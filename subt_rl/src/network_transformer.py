
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from custom_transformer import TransformerEncoder, TransformerEncoderLayer
from custom_transformer import TransformerDecoder, TransformerDecoderLayer


class RadarTransformer(nn.Module):
    def __init__(
        self,
        features=7,
        embed_dim=64,
        nhead=8,
        encoder_layers=6,
        decoder_layers=6,
    ):
        super(RadarTransformer, self).__init__()

        # linear projection layer
        self.linear_projection = nn.Linear(features, embed_dim)

        # transformer encder block
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer,
            num_layers=encoder_layers
        )

        # decoder queries
        self.quries = nn.Parameter(torch.randn(241, 1, embed_dim))

        # transformer decoder layer
        decoder_layer = TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
        )
        self.transformer_decoder = TransformerDecoder(
            decoder_layer,
            num_layers=decoder_layers,
        )

        # linear to output ranges
        self.linear_to_range = nn.Linear(embed_dim, 1)

    def forward(self, x, pad_mask, attention_map=False):
        n, b, _ = x.shape

        x = self.linear_projection(x)

        # transformer encoder
        x, ecd_att_map = self.transformer_encoder(
            x,
            src_key_padding_mask=pad_mask
        )
        # print(ecd_att_map.shape)

        # generate queries
        qur = self.quries.repeat(1, b, 1)

        # transformer decoder
        decoded, dcd_att_map = self.transformer_decoder(
            tgt=qur,
            memory=x,
            memory_key_padding_mask=pad_mask,
        )

        # project to ranges
        ranges = self.linear_to_range(decoded)
        ranges = torch.squeeze(ranges)
        if b>1:
            ranges = torch.transpose(ranges, 0, 1)

        if attention_map:
            return ranges, ecd_att_map, dcd_att_map
        else:
            return ranges


class RadarEncoder(nn.Module):
    def __init__(
        self,
        features=7,
        embed_dim=64,
        nhead=8,
        layers=6,
        output_embeding=512,
    ):
        super(RadarEncoder, self).__init__()
        # linear projection layer
        self.linear_projection = nn.Linear(features, embed_dim)

        # decoder queries
        self.quries = nn.Parameter(torch.randn(output_embeding, 1, embed_dim))

        # transformer decoder layer
        decoder_layer = TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
        )
        self.transformer_decoder = TransformerDecoder(
            decoder_layer,
            num_layers=layers,
        )

        # linear to output ranges
        self.linear_to_output = nn.Linear(embed_dim, 1)

    def forward(self, x, pad_mask, need_attention_map=False):
        n, b, _ = x.shape

        x = self.linear_projection(x)

        # generate queries
        qur = self.quries.repeat(1, b, 1)

        # transformer decoder
        decoded, dcd_att_map = self.transformer_decoder(
            tgt=qur,
            memory=x,
            memory_key_padding_mask=pad_mask,
        )

        # project to feature embedding
        embed = self.linear_to_output(decoded)
        embed = torch.squeeze(embed)
        
        if b>1:
            embed = torch.transpose(embed, 0, 1)

        if need_attention_map:
            return embed, dcd_att_map
        else:
            return embed
