import torch
import torch.nn.functional as F
import typing
import logging
import pytorch_lightning as pl

# from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from einops import repeat, rearrange
from torch import nn

from mae.evaluation import Lightning_Eval
from mae.utils import CosineLinearWarmupScheduler


class MAE(pl.LightningModule):
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
        # save hyperparameters for easy inference
        self.save_hyperparameters(
            ignore=["encoder", "decoder", "decoder_pos_emb", "enc2dec", "to_pixels" "mask_token"]
        )

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

        # Save constants
        pixel_values_per_patch = self.encoder.patch_size**2 * self.encoder.in_chans
        encoder.dim = encoder.embed_dim
        self.num_patches = self.encoder.patch_embed.num_patches

        self.decoder = decoder.to(self.device)

        # Projection to match encoder/decoder dimensions
        self.enc2dec = nn.Linear(encoder.dim, decoder.dim, bias=True)

        # Initialize decoder
        # Masking tokens for decoder.
        self.mask_token = nn.Parameter(torch.randn(decoder.dim))

        # Fixed embeddings for decoder
        self.decoder_pos_emb = nn.Embedding(self.num_patches, decoder.dim)

        self.to_pixels = nn.Linear(decoder.dim, pixel_values_per_patch)

    def forward(self, x):
        """
        Encode image.

        Args:
            x: Image to encode

        Returns:
            Encoded image
        """
        # dimension (batch, features), features from config e.g. 512
        return self.encoder(x)

    def img_to_reconstruction(self, x):
        # Patch to encoder tokens and add positions
        tokens = self.encoder.patch_embed(x)

        batch, num_patches, *_ = tokens.shape

        tokens = tokens + self.encoder.pos_embed[:, 1 : (num_patches + 1)]

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

    # def patchify(self, imgs):
    #     """
    #     imgs: (N, 3, H, W)
    #     x: (N, L, patch_size**2 *3)
    #     """
    #     p = self.encoder.patch_embed.patch_size[0]

    #     assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    #     h = w = imgs.shape[2] // p
    #     x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    #     x = torch.einsum("nchpwq->nhwpqc", x)
    #     x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
    #     return x

    def patchify(self, imgs):
        """
        imgs: (N, C, H, W)
        x: (N, L, patch_size**2 *C)
        """
        p = self.encoder.patch_embed.patch_size[0]

        # Turn (c x h x w) images into (n_patchs x patch_dim) flattened patches and
        # project to latent dimension
        x = rearrange(imgs, "b c (n1 p1) (n2 p2) -> b (n1 n2) (p1 p2 c)", p1=p, p2=p)
        return x

    # def unpatchify(self, x):
    #     """
    #     x: (N, L, patch_size**2 *3)
    #     imgs: (N, 3, H, W)
    #     """
    #     p = self.encoder.patch_embed.patch_size[0]
    #     h = w = int(x.shape[1] ** 0.5)
    #     logging.info(f"{h}, {w}, {x.shape}")
    #     assert h * w == x.shape[1]

    #     x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    #     x = torch.einsum("nhwpqc->nchpwq", x)
    #     imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    #     return imgs

    def training_step(self, batch, batch_idx):
        """
        Training step for MAE model.

        Args:
            batch: Batch of data
            batch_idx: Batch index

        Returns:
            Loss for training step
        """
        # Get patches from image
        x, _ = batch

        # Get batch size and number of patches
        patches = self.patchify(x)
        batch, num_patches, *_ = patches.shape
        # logging.info(f"\n {patches.shape} \n")

        # Patch to encoder tokens and add positions
        tokens = self.encoder.patch_embed(x)
        # logging.info(f"\n {tokens.shape} \n")

        tokens = tokens + self.encoder.pos_embed[:, 1 : (num_patches + 1)]

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
        encoded_tokens = self.encoder.transform(tokens)

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
        # pred_pixel_values = self.unpatchify(masked_pixels)

        # calculate reconstruction loss
        loss = F.mse_loss(pred_pixel_values, masked_patches)

        # scaled recon loss
        # loss = F.mse_loss(pred_pixel_values, masked_patches, reduction="none")
        # min_value = torch.min(masked_patches)
        # scaling_mask = torch.ones_like(loss)
        # scaling_mask[torch.argwhere(masked_patches == min_value)] = 0.2

        self.log("train/loss", loss, on_step=True, on_epoch=True)
        return loss

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
