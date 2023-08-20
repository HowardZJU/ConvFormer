import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

ACT2FN = {
        "gelu": lambda x: x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0))), 
        "relu": F.relu, 
        "swish": lambda x: x * torch.sigmoid(x)}


class SelfAttention(nn.Module):
    def __init__(self, args):
        super(SelfAttention, self).__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (args.hidden_size, args.num_attention_heads))
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)

        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class FastSelfAttention(SelfAttention):
    def __init__(self, args):
        super().__init__(args)

        self.query_att = nn.Linear(self.all_head_size, self.num_attention_heads)
        self.key_att = nn.Linear(self.all_head_size, self.num_attention_heads)
        self.transform = nn.Linear(self.all_head_size, self.all_head_size)

    def forward(self, input_tensor, attention_mask):
        """
        Following the open source code https://github.com/wuch15/Fastformer/blob/main/Fastformer.ipynb
        """
        batch, seq_len, _ = input_tensor.shape
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        query_for_score = self.query_att(mixed_query_layer).transpose(1, 2) / (self.attention_head_size ** 0.5) # batch, num_head, seq_len
        query_for_score += attention_mask
        query_weight = nn.Softmax(dim=-1)(query_for_score).unsqueeze(2)
        # dropout?

        query_layer = self.transpose_for_scores(mixed_key_layer)
        pooled_query = torch.matmul(query_weight, query_layer).transpose(1, 2).view(-1,1,self.num_attention_heads*self.attention_head_size)
        pooled_query_repeat= pooled_query.repeat(1, seq_len,1) # batch_size, num_head, seq_len, head_dim
        mixed_query_key_layer=mixed_key_layer * pooled_query_repeat
        
        query_key_score=(self.key_att(mixed_query_key_layer)/ self.attention_head_size**0.5).transpose(1, 2)
        query_key_score += attention_mask
        query_key_weight = nn.Softmax(dim=-1)(query_key_score).unsqueeze(2)

        key_layer = self.transpose_for_scores(mixed_query_key_layer)
        pooled_key = torch.matmul(query_key_weight, key_layer)

        #query = value
        weighted_value =(pooled_key * query_layer).transpose(1, 2)
        weighted_value = weighted_value.reshape(
            weighted_value.size()[:-2] + (self.num_attention_heads * self.attention_head_size,))
        weighted_value = self.out_dropout(weighted_value)
        # hidden_states = self.LayerNorm(self.transform(weighted_value) + mixed_key_layer)
        hidden_states = self.transform(weighted_value) + mixed_key_layer

        return hidden_states
    

class NoSharedSelfAttention(SelfAttention):
    def __init__(self, args):
        super().__init__(args)
        del self.query
        del self.key
        del self.value
        self.query = [nn.Linear(args.hidden_size, self.all_head_size).to('cuda:0') for i in range(args.max_seq_length)]
        self.key = [nn.Linear(args.hidden_size, self.all_head_size).to('cuda:0') for i in range(args.max_seq_length)]
        self.value = [nn.Linear(args.hidden_size, self.all_head_size).to('cuda:0') for i in range(args.max_seq_length)]

    def forward(self, input_tensor, attention_mask):
        # batch, seq, hidden

        mixed_query_layer = torch.stack([self.query[i](input_tensor[:, i, :]) for i in range(input_tensor.shape[1])], dim=1)
        mixed_key_layer = torch.stack([self.key[i](input_tensor[:, i, :]) for i in range(input_tensor.shape[1])], dim=1)
        mixed_value_layer = torch.stack([self.value[i](input_tensor[:, i, :]) for i in range(input_tensor.shape[1])], dim=1)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class OverfitSelfAttention(SelfAttention):
    def __init__(self, args):
        super().__init__(args)
        del self.query
        del self.key
        del self.value
        self.query = nn.Linear(args.hidden_size*args.max_seq_length, self.all_head_size*args.max_seq_length)
        self.key = nn.Linear(args.hidden_size*args.max_seq_length, self.all_head_size*args.max_seq_length)
        self.value = nn.Linear(args.hidden_size*args.max_seq_length, self.all_head_size*args.max_seq_length)

    def forward(self, input_tensor, attention_mask):
        batch, seq, hidden = input_tensor.shape
        mixed_query_layer = self.query(torch.flatten(input_tensor, start_dim=1)).reshape(batch, seq, hidden)
        mixed_key_layer = self.key(torch.flatten(input_tensor, start_dim=1)).reshape(batch, seq, hidden)
        mixed_value_layer = self.value(torch.flatten(input_tensor, start_dim=1)).reshape(batch, seq, hidden)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_scores = attention_scores

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class LocalSelfAttention(SelfAttention):
    def __init__(self, args):
        super().__init__(args)
        self.local_mask = nn.Parameter(
            (
                torch.ones(args.max_seq_length, args.max_seq_length) - 
                torch.triu(torch.ones(args.max_seq_length, args.max_seq_length), args.local) - 
                torch.tril(torch.ones(args.max_seq_length, args.max_seq_length), -args.local)
            ), 
            requires_grad=False)
        self.local_mask = nn.Parameter(
            (
                torch.triu(torch.ones(args.max_seq_length, args.max_seq_length), args.local) +
                torch.tril(torch.ones(args.max_seq_length, args.max_seq_length), -args.local)
            ) * -1e4, 
            requires_grad=False)
        

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        mask = torch.minimum(attention_mask, self.local_mask)
        attention_scores = attention_scores + mask
        attention_scores = attention_scores

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # attention_probs = attention_probs * self.local_mask
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states



class PoolingSelfAttention(SelfAttention):
    def __init__(self, args):
        super().__init__(args)
        self.local_mask = nn.Parameter(
            (
                torch.triu(torch.ones(args.max_seq_length, args.max_seq_length), args.local) +
                torch.tril(torch.ones(args.max_seq_length, args.max_seq_length), -args.local)
            ) * -1e4, 
            requires_grad=False)
        self.query_2 = nn.Linear(args.hidden_size, self.all_head_size)
        self.key_2 = nn.Linear(args.hidden_size, self.all_head_size)
        self.value_2 = nn.Linear(args.hidden_size, self.all_head_size)
        self.pool = nn.MaxPool1d(kernel_size=args.pool_size, stride=args.pool_size)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # mask = torch.minimum(attention_mask, self.local_mask)
        mask = self.local_mask
        attention_scores = attention_scores + mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        output_1 = context_layer.view(*new_context_layer_shape)


        mixed_query_layer = self.query_2(output_1)
        mixed_key_layer = self.pool(self.key_2(output_1).transpose(1, 2)).transpose(1, 2)
        mixed_value_layer = self.pool(self.value_2(output_1).transpose(1, 2)).transpose(1, 2)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        hidden_states = context_layer
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class ConvLayer(nn.Module):
    def __init__(self, args):
        super(ConvLayer, self).__init__()
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.conv = nn.Sequential()
        self.padding_len = args.conv_size - 1
        if args.padding_mode == 0:
            self.padding_mode = 'circular'
        elif args.padding_mode == 1:
            self.padding_mode = 'reflect'
        elif args.padding_mode == 2:
            self.padding_mode = 'constant'
        self.conv_size = args.conv_size

        if args.conv_name == 0 or args.ablate == 4:
            self.conv.add_module('conv', nn.Conv1d(args.hidden_size, args.hidden_size, kernel_size=self.conv_size))
        elif args.conv_name == 1:
            conv = nn.Conv1d(args.hidden_size, args.hidden_size, kernel_size=self.conv_size, groups=args.hidden_size)
            if args.initialize == 1:
                init_ratio = 5e-3
                conv.weight.data.normal_(0.0, init_ratio)
                conv.bias.data.normal_(0.0, init_ratio)
            self.conv.add_module('depthwise_conv', conv)
        elif args.conv_name == 2:
            self.conv.add_module('depthwise_conv', nn.Conv1d(args.hidden_size, args.hidden_size, kernel_size=self.conv_size, groups=args.hidden_size))
            self.conv.add_module('pointwise_conv', nn.Conv1d(args.hidden_size, args.hidden_size, kernel_size=1))
        
        self.act_fn = ACT2FN[args.hidden_act]
        self.act = args.act
        self.args = args

    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        batch, seq_len, hidden = input_tensor.shape
        x = input_tensor.transpose(1, 2)
        x = nn.functional.pad(x, (self.padding_len, 0), self.padding_mode)
        x = self.conv(x).transpose(1, 2)
        x = self.act_fn(x) if self.args.ablate == 5 else x
        hidden_states = self.out_dropout(x)
        if self.args.ablate == 2:
            hidden_states = self.LayerNorm(hidden_states)
        elif self.args.ablate == 3:
            hidden_states = hidden_states + input_tensor
        elif self.args.ablate == 7:
            hidden_states = hidden_states
        else:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class FConvLayer(nn.Module):
    def __init__(self, args):
        super(FConvLayer, self).__init__()
        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.conv_weight = nn.Parameter(torch.randn(1, args.conv_size, args.hidden_size, dtype=torch.float32) * 0.02)
        self.zeros = nn.Parameter(torch.zeros(1, args.max_seq_length-args.conv_size, args.hidden_size), requires_grad=False)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.no_filter = args.no_filter
        self.args = args


    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        if self.no_filter is False:
            # padding should be conducted on the right, since fconv is equal to the vanilla convolution with flip.
            weight = torch.cat([self.conv_weight, self.zeros], dim=1) 
            weight = torch.fft.rfft(weight, dim=1, norm='ortho')
            x = x * weight
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm='ortho')
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class SynthesisV1(nn.Module):
    """
    This is a MHSA with single head for fair comparison
    """
    def __init__(self, args):
        super(SynthesisV1, self).__init__()

        self.query = nn.Linear(args.hidden_size, args.hidden_size)
        self.key = nn.Linear(args.hidden_size, args.hidden_size)
        self.value = nn.Linear(args.hidden_size, args.hidden_size)

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)


    def forward(self, input_tensor, attention_mask):
        query = self.query(input_tensor)
        key = self.key(input_tensor)
        value = self.value(input_tensor)

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(query.shape[-1])
        attention_scores = attention_scores + attention_mask.squeeze(1)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class SynthesisV11(SynthesisV1):
    """
    This is a MHSA with q=k
    """
    def __init__(self, args):
        super(SynthesisV11, self).__init__(args)
        del self.key

    def forward(self, input_tensor, attention_mask):
        query = self.query(input_tensor)
        key = self.query(input_tensor)
        value = self.value(input_tensor)

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(query.shape[-1])
        attention_scores = attention_scores + attention_mask.squeeze(1)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class SynthesisV12(SynthesisV1):
    """
    This is a MHSA with q=k=v
    """
    def __init__(self, args):
        super(SynthesisV12, self).__init__(args)
        del self.key
        del self.value

    def forward(self, input_tensor, attention_mask):
        query = self.query(input_tensor)
        key = self.query(input_tensor)
        value = self.query(input_tensor)

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(query.shape[-1])
        attention_scores = attention_scores + attention_mask.squeeze(1)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class SynthesisV2(nn.Module):
    """
    This is Synthesiser-D
    """
    def __init__(self, args):
        super(SynthesisV2, self).__init__()

        self.attn_scores = nn.Linear(args.hidden_size, args.max_seq_length)
        self.value = nn.Linear(args.hidden_size, args.hidden_size)

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)


    def forward(self, input_tensor, attention_mask):

        value = self.value(input_tensor)
        attention_scores = self.attn_scores(input_tensor)
        attention_scores = attention_scores / math.sqrt(value.shape[-1])
        attention_scores = attention_scores + attention_mask.squeeze(1)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class SynthesisV3(nn.Module):
    """
    This is Synthesiser-R
    """
    def __init__(self, args):
        super(SynthesisV3, self).__init__()

        self.attn_scores = nn.Parameter(torch.randn(1, args.max_seq_length, args.max_seq_length, dtype=torch.float32)*0.02)
        self.value = nn.Linear(args.hidden_size, args.hidden_size)

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)


    def forward(self, input_tensor, attention_mask):
        
        value = self.value(input_tensor)
        attention_scores = self.attn_scores + attention_mask.squeeze(1)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class SynthesisV4(nn.Module):
    """
    This is Synthesiser-R with non-trainable parameters.
    """
    def __init__(self, args):
        super(SynthesisV4, self).__init__()

        self.attn_scores = nn.Parameter(torch.randn(1, args.max_seq_length, args.max_seq_length, dtype=torch.float32)*0.02, requires_grad=False)
        self.value = nn.Linear(args.hidden_size, args.hidden_size)

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)


    def forward(self, input_tensor, attention_mask):
        
        value = self.value(input_tensor)
        attention_scores = self.attn_scores + attention_mask.squeeze(1)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class SynthesisV5(nn.Module):
    """
    This is a dense layer with meta-former structure
    """
    def __init__(self, args):
        super(SynthesisV5, self).__init__()

        self.attn_scores = nn.Parameter(torch.randn(1, args.max_seq_length, args.max_seq_length, dtype=torch.float32)*0.02, requires_grad=False)
        self.value = nn.Linear(args.hidden_size, args.hidden_size)

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)


    def forward(self, input_tensor, attention_mask):
        
        value = self.value(input_tensor)
        hidden_states = self.dense(value)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class Intermediate(nn.Module):
    def __init__(self, args):
        super(Intermediate, self).__init__()
        self.dense_1 = nn.Linear(args.hidden_size, args.hidden_size * args.ffn_multiplier)
        if isinstance(args.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[args.hidden_act]
        else:
            self.intermediate_act_fn = args.hidden_act

        self.dense_2 = nn.Linear(args.ffn_multiplier * args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

    def forward(self, input_tensor):
        if self.args.ablate == 6:
            return input_tensor
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_2(hidden_states) 
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Layer(nn.Module):
    def __init__(self, args):
        super(Layer, self).__init__()

        self.model_name = args.model_name
        if self.model_name == 'SASRec':
            self.attention = SelfAttention(args)
        elif self.model_name == 'SyntV1':
            self.attention = SynthesisV1(args)
        elif self.model_name == 'SyntV11':
            self.attention = SynthesisV11(args)
        elif self.model_name == 'SyntV12':
            self.attention = SynthesisV12(args)
        elif self.model_name == 'SyntV2':
            self.attention = SynthesisV2(args)
        elif self.model_name == 'SyntV3':
            self.attention = SynthesisV3(args)
        elif self.model_name == 'SyntV4':
            self.attention = SynthesisV4(args)
        elif self.model_name == 'SyntV5':
            self.attention = SynthesisV5(args)
        elif self.model_name == 'LocalSelfAttn': # for ablation
            self.attention = LocalSelfAttention(args)
        elif self.model_name == 'FastSelfAttn':
            self.attention = FastSelfAttention(args)
        elif self.model_name == 'PoolingSelfAttn':
            self.attention = PoolingSelfAttention(args)
        elif self.model_name == 'NoSharedSelfAttn': # for ablation
            self.attention = NoSharedSelfAttention(args)
        elif self.model_name == 'OverfitSelfAttn': # for ablation
            self.attention = OverfitSelfAttention(args)
        elif self.model_name == 'CONV':
            self.filterlayer = ConvLayer(args)
        elif self.model_name == 'FCONV':
            self.filterlayer = FConvLayer(args)
        self.intermediate = Intermediate(args)

    def forward(self, hidden_states, attention_mask):
        if (self.model_name == 'SASRec') or ('Attn' in self.model_name) or ('Synt' in self.model_name):
            hidden_states = self.attention(hidden_states, attention_mask)
        else:
            hidden_states = self.filterlayer(hidden_states)

        intermediate_output = self.intermediate(hidden_states)
        return intermediate_output

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        layer = Layer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers
