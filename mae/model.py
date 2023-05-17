import torch
import torch.nn.functional as F
import typing

# from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from einops import repeat
from torch import nn

from mae.evaluation import Lightning_Eval
from mae.utils import CosineLinearWarmupScheduler


class MAE(Lightning_Eval):
    def __init__(
        self,
        encoder,
        decoder,
        n_epochs: int = 600,
        warmup_epochs: int = 10,
        masking_ratio: float = 0.75,
        lr: float = 0.2,
        momentum: float = 0.9,
        weight_decay: float = 0.0000015,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
    ):
        """
        Create a Masked Autoencoder (MAE) model for training with Pytorch Lightning. Inherit linear
        evaluation from the Lightning_Eval class.

        :param encoder: Encoder network
        :param decoder: Decoder network
        :param masking_ratio: Masking ratio for MAE - controls the proportion of patches masked
        :param lr: Learning rate for training
        :param momentum: Momentum for training
        :param weight_decay: Weight decay (optimizer)
        :param beta_1: Beta 1 (optimizer)
        :param beta_2: Beta 2 (optimizer)
        :param n_epochs: Number of epochs to train for
        :param warmup_epochs: Number of epochs to warmup for
        """
        super().__init__()
        self.save_hyperparameters()  # save hyperparameters for easy inference

        self.masking_ratio = masking_ratio
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.n_epochs = n_epochs
        self.warmup_epochs = warmup_epochs

        # Initialize encoder
        self.encoder = encoder.to(self.device)

        # Get patch mappings from encoder
        self.to_patch, self.patch_to_emb = self.encoder.to_patch_embedding[:2]
        pixel_values_per_patch = self.patch_to_emb.weight.shape[-1]
        num_patches, encoder.dim = self.encoder.pos_embedding.shape[-2:]

        self.decoder = decoder.to(self.device)

        # Projection to match encoder/decoder dimensions
        self.enc2dec = (
            nn.Linear(encoder.dim, decoder.dim) if encoder.dim != decoder.dim else nn.Identity()
        )

        # Initialize decoder
        # Masking tokens for decoder.
        self.mask_token = nn.Parameter(torch.randn(decoder.dim))

        # Fixed embeddings for decoder
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder.dim)
        self.to_pixels = nn.Linear(decoder.dim, pixel_values_per_patch)

    def forward(self, x):
        """
        Encode image.

        :param x: Image

        :return: Encoded image
        """
        return self.encoder(x)  # dimension (batch, features), features from config e.g. 512

    def img_to_reconstruction(self, x):
        # Get patches from image
        patches = self.to_patch(x)

        # Get batch size and number of patches
        batch, num_patches, *_ = patches.shape

        # Patch to encoder tokens and add positions
        tokens = self.patch_to_emb(patches)
        tokens = tokens + self.encoder.pos_embedding[:, 1 : (num_patches + 1)]

        # Calculate number of patches to mask
        num_masked = int(self.masking_ratio * num_patches)

        # Get random indices to choose random masked patches
        rand_indices = torch.rand(batch, num_patches).argsort(dim=-1).to(self.device)

        # Save masked and unmasked indices
        masked_indices, unmasked_indices = (
            rand_indices[:, :num_masked],
            rand_indices[:, num_masked:],
        )

        # Get the unmasked tokens to be encoded
        batch_range = torch.arange(batch)[:, None]
        tokens = tokens[batch_range, unmasked_indices]

        # Get the patches to be masked for the final reconstruction loss
        masked_patches = patches[batch_range, masked_indices]

        # Attend with vision transformer
        encoded_tokens = self.encoder.transformer(tokens)

        # Project encoder to decoder dimensions, if they are not equal,
        # the paper says you can get away with a smaller dimension for decoder
        decoder_tokens = self.enc2dec(encoded_tokens)

        # Reapply decoder position embedding to unmasked tokens
        decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)

        # Repeat mask tokens for number of masked, and add the positions
        # using the masked indices derived above
        mask_tokens = repeat(self.mask_token, "d -> b n d", b=batch, n=num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        # Concat the masked tokens to the decoder tokens and attend with decoder
        decoder_tokens = torch.cat((mask_tokens, decoder_tokens), dim=1)
        decoded_tokens = self.decoder(decoder_tokens)

        # Splice out the mask tokens
        mask_tokens = decoded_tokens[:, :num_masked]

        # Project to pixel values
        pred_pixel_values = self.to_pixels(mask_tokens)

    def training_step(self, batch, batch_idx):
        """
        Training step for MAE model.

        :param batch: Batch of data
        :param batch_idx: Batch index
        :return: Loss for training step
        """
        # Get patches from image
        x, _ = batch

        patches = self.to_patch(x)

        # Get batch size and number of patches
        batch, num_patches, *_ = patches.shape

        # Patch to encoder tokens and add positions
        tokens = self.patch_to_emb(patches)
        tokens = tokens + self.encoder.pos_embedding[:, 1 : (num_patches + 1)]

        # Calculate number of patches to mask
        num_masked = int(self.masking_ratio * num_patches)

        # Get random indices to choose random masked patches
        rand_indices = torch.rand(batch, num_patches).argsort(dim=-1).to(self.device)

        # Save masked and unmasked indices
        masked_indices, unmasked_indices = (
            rand_indices[:, :num_masked],
            rand_indices[:, num_masked:],
        )

        # Get the unmasked tokens to be encoded
        batch_range = torch.arange(batch)[:, None]
        tokens = tokens[batch_range, unmasked_indices]

        # Get the patches to be masked for the final reconstruction loss
        masked_patches = patches[batch_range, masked_indices]

        # Attend with vision transformer
        encoded_tokens = self.encoder.transformer(tokens)

        # Project encoder to decoder dimensions, if they are not equal,
        # the paper says you can get away with a smaller dimension for decoder
        decoder_tokens = self.enc2dec(encoded_tokens)

        # Reapply decoder position embedding to unmasked tokens
        decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)

        # Repeat mask tokens for number of masked, and add the positions
        # using the masked indices derived above
        mask_tokens = repeat(self.mask_token, "d -> b n d", b=batch, n=num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        # Concat the masked tokens to the decoder tokens and attend with decoder
        decoder_tokens = torch.cat((mask_tokens, decoder_tokens), dim=1)
        decoded_tokens = self.decoder(decoder_tokens)

        # Splice out the mask tokens
        mask_tokens = decoded_tokens[:, :num_masked]

        # Project to pixel values
        pred_pixel_values = self.to_pixels(mask_tokens)

        # calculate reconstruction loss
        loss = F.mse_loss(pred_pixel_values, masked_patches)

        # scaled recon loss
        # loss = F.mse_loss(pred_pixel_values, masked_patches, reduction="none")
        # min_value = torch.min(masked_patches)
        # scaling_mask = torch.ones_like(loss)
        # scaling_mask[torch.argwhere(masked_patches == min_value)] = 0.2

        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss

    @property
    def backbone(self):
        return self.encoder

    def configure_optimizers(self):
        params = self.parameters()

        opt = torch.optim.AdamW(
            params,
            lr=self.lr,
            betas=(self.beta_2, self.beta_2),
            weight_decay=self.weight_decay,
        )

        scheduler = CosineLinearWarmupScheduler(
            opt, warmup_epochs=self.warmup_epochs, max_epochs=self.n_epochs
        )
        return [opt], [scheduler]
