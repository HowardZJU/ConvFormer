import torch
import torch.nn as nn
from modules import Encoder, LayerNorm

def reverse_non_zero_elements(batch):
    # Create a mask for non-zero elements
    non_zero_mask = batch != 0
    batch_reverse = batch.clone()

    # Iterate over each sequence in the batch
    for i in range(batch.size(0)):
        # Extract non-zero elements
        non_zero_elements = batch[i][non_zero_mask[i]]

        # Reverse the non-zero elements
        reversed_elements = non_zero_elements.flip(dims=[0])

        # Place the reversed elements back into the original tensor
        batch_reverse[i][non_zero_mask[i]] = reversed_elements

    return batch_reverse

def switch_non_zero_subsections(batch):
    batch_reverse = batch.clone()
    for i in range(batch.size(0)):
        # Create a mask and extract non-zero elements
        non_zero_mask = batch[i] != 0
        non_zero_elements = batch[i][non_zero_mask]

        # Calculate the midpoint for the non-zero elements
        midpoint = non_zero_elements.size(0) // 2

        # Partition and switch the order of non-zero elements
        first_half = non_zero_elements[:midpoint]
        second_half = non_zero_elements[midpoint:]
        switched_elements = torch.cat((second_half, first_half), dim=0)

        # Place the switched elements back into the original sequence
        batch_reverse[i][non_zero_mask] = switched_elements

    return batch_reverse

class OurRecommender(nn.Module):
    def __init__(self, args):
        super(OurRecommender, self).__init__()
        self.args = args
        self.zero_pad = args.zero_pad
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        # self.item_embeddings.retain_grad()
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = Encoder(args)
        # self.item_encoder = VGG16()
        # self.scale_agg = nn.Linear(args.hidden_size * args.num_hidden_layers, args.hidden_size)
        self.use_causalmask = args.use_causalmask
        self.switch_order = args.switch_order
        self.apply(self.init_weights)
        
    def add_position_embedding(self, sequence):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        if self.args.ablate == 1:
            sequence_emb = item_embeddings
        else:
            sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    # same as SASRec
    def forward(self, input_ids):
        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape, device=input_ids.device), diagonal=1) # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        if self.use_causalmask == 1:
            extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        if self.args.model_name == 'FastSelfAttn':
            extended_attention_mask = attention_mask.unsqueeze(1) # torch.int64
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        if self.switch_order == 1:
            input_ids = reverse_non_zero_elements(input_ids)
        if self.switch_order == 2:
            input_ids = switch_non_zero_subsections(input_ids)
        sequence_emb = self.add_position_embedding(input_ids)
        if self.zero_pad == 1:
            sequence_emb = sequence_emb * attention_mask.unsqueeze(2)
        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True,
                                                )
        # multiscale
        # if self.args.multiscale == 0:
        sequence_output = item_encoded_layers[-1]
        # elif self.args.multiscale == 1:
        #     sequence_output = sum(item_encoded_layers) / self.args.num_hidden_layers
        # elif self.args.multiscale == 2:
        #     # sequence_output = torch.stack(item_encoded_layers).max(0)[0]
        #     sequence_output = torch.stack(item_encoded_layers, dim=2).flatten(2)
        #     sequence_output = self.scale_agg(sequence_output)
        
        return sequence_output

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
