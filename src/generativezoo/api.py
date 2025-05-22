from fastapi import FastAPI, HTTPException
from loguru import logger
import torch
from data.CycleGAN_Dataloaders import *
from config import data_raw_dir
from utils.util import parse_args_VanillaSGM
from data.Dataloaders import pick_dataset
from models.SGM.VanillaSGM import VanillaSGM
from models.SGM.NCSNv2 import NCSNv2  
from models.VAE.VanillaVAE import VanillaVAE
from models.VAE.HierarchicalVAE import HierarchicalVAE
from models.VAE.ConditionalVAE import ConditionalVAE
from models.DDPM.ConditionalDDPM import ConditionalDDPM
from models.DDPM.MONAI_DiffAE import DiffAE
from models.DDPM.VanillaDDPM import VanillaDDPM
from models.AR.PixelCNN import PixelCNN
from models.AR.VQGAN_Transformer import VQGANTransformer
from models.AR.VQVAE_Transformer import VQVAETransformer
from models.GAN.AdversarialVAE import AdversarialVAE
from models.GAN.VanillaGAN import VanillaGAN
from models.GAN.ConditionalGAN import ConditionalGAN
from models.GAN.CycleGAN import CycleGAN
from models.GAN.PrescribedGAN import PresGAN
from models.GAN.WGAN import WGAN
from models.Flow.VanillaFlow import VanillaFlow
from models.Flow.RealNVP import RealNVP
from models.Flow.Glow import Glow
from models.Flow.FlowPlusPlus import FlowPlusPlus
from models.Flow.FlowMatching import FlowMatching
from models.Flow.CondFlowMatching import CondFlowMatching
from models.Flow.RectifiedFlows import RF
from model import *
from models.GAN.VanillaGAN import *
from fastapi.responses import JSONResponse
from typing import Union, Literal
from pydantic import ValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import os
import shutil
import tempfile
import yaml

app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,  
    allow_methods=["*"],  
    allow_headers=["*"],  
)


@app.post("/sgm_train")
async def train_model(request: dict):  # Raw JSON data is received
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Selected device: {device}")

    try:
        # Convert JSON content into the appropriate model based on `model_name`
        if request["model_name"] == "Vanilla SGM":
            try:
                validated_request = VanillaSGMRequestTrain(**request)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid parameters for Vanilla SGM: {e}")

            logger.info(f"Loading dataset for Vanilla SGM: {validated_request.dataset}")
            dataloader, input_size, channels = pick_dataset(
                validated_request.dataset,
                'train',
                validated_request.batch_size,
                normalize=True,
                size=None,
                num_workers=validated_request.num_workers
            )
            logger.info("Dataset successfully loaded.")

            model = VanillaSGM(validated_request, channels, input_size)
            logger.info("Vanilla SGM training is starting.")
            model.train_model(dataloader)
            logger.info("Vanilla SGM training completed.")

            return JSONResponse(status_code=200, content={"status": "Success", "message": "Vanilla SGM training completed successfully."})

        elif request["model_name"] == "NCSNv2":
            try:
                validated_request = NCSNv2RequestTrain(**request)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid parameters for NCSNv2: {e}")

            logger.info(f"Loading dataset for NCSNv2: {validated_request.dataset}")
            logger.info(f"Request details: {validated_request}")
            train_loader, input_size, channels = pick_dataset(
                dataset_name=validated_request.dataset,
                batch_size=validated_request.batch_size,
                normalize=False,
                size=None,
                num_workers=validated_request.num_workers
            )
            logger.info("Dataset successfully loaded.")

            model = NCSNv2(input_size, channels, validated_request)
            logger.info("NCSNv2 training is starting.")
            model.train_model(train_loader, validated_request)
            logger.info("NCSNv2 training completed.")

            return JSONResponse(status_code=200, content={"status": "Success", "message": "NCSNv2 training completed successfully."})

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model: {request['model_name']}. Supported models: Vanilla SGM, NCSNv2."
            )

    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during model training: {str(e)}"
        )

                               

@app.post("/sgm_sample")
async def sample_model(request: dict):
    try:
        
        if request["model_name"] == "Vanilla SGM":
            try:
                validated_request = VanillaSGMRequestSample(**request)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid parameters for Vanilla SGM: {e}")

            logger.info(f"Loading dataset: {validated_request.dataset}")
            _, input_size, channels = pick_dataset(
                validated_request.dataset,
                'val',
                validated_request.batch_size,
                normalize=True,
                size=None
            )
            data_set= validated_request.dataset
            print("data set check control:", validated_request.dataset)
            logger.info("Dataset successfully loaded.")

            logger.info(f"API parameters: {validated_request.dict()}")

            model = VanillaSGM(validated_request, channels, input_size)
            checkpoint = f"/home/a30/Desktop/zoo/models/VanillaSGM/VanSGM_{data_set}.pt"
            model.model.load_state_dict(torch.load(checkpoint))
            logger.info("Checkpoint successfully loaded.")

            # Sampling process
            logger.info(f"Generating {validated_request.num_samples} samples.")
            base64_image = model.sample(validated_request.num_samples)

            if not base64_image:
                raise ValueError("Sampling process failed. No Base64 image generated.")

            return {
                "status": "success",
                "message": "Samples generated successfully!",
                "base64_image": base64_image
            }

        elif request["model_name"] == "NCSNv2":
            try:
                validated_request = NCSNv2RequestSample(**request)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid parameters for NCSNv2: {e}")
            logger.info("NCSNv2 is currently not supported for sampling.")
            _, input_size, channels = pick_dataset(
                dataset_name=validated_request.dataset, 
                batch_size=validated_request.batch_size, 
                normalize=False, 
                size=None
            )
            data_set= validated_request.dataset
            logger.info("Dataset successfully loaded.")
            print("data set check control:", validated_request.dataset)
            logger.info(f"API parameters: {validated_request.dict()}")
            model = NCSNv2(input_size, channels, validated_request)
            checkpoint = f"/home/a30/Desktop/zoo/models/NCSNv2/NCSNv2_{data_set}.pt"
            model.model.load_state_dict(torch.load(checkpoint))

            logger.info("Checkpoint successfully loaded.")
           # Sampling process
            logger.info(f"Generating {validated_request.num_samples} samples.")
            base64_image = model.sample(validated_request)

            if not base64_image:
                raise ValueError("Sampling process failed. No Base64 image generated.")

            return {
                "status": "success",
                "message": "Samples generated successfully!",
                "base64_image": base64_image
            }
   
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model: {request['model_name']}. Supported models: Vanilla SGM, NCSNv2."
            )

    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during sampling: {str(e)}"
        )
@app.post("/vanillavae_train")
async def train_model(request: dict):  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Selected device: {device}")

    try:
        # Convert JSON content into the appropriate model based on `model_name`
        if request["model_name"] == "Vanilla VAE":
            try:
                validated_request = VanillaVAERequestTrain(**request)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid parameters for Vanilla SGM: {e}")

            logger.info(f"Loading dataset for Vanilla SGM: {validated_request.dataset}")

            train_loader, in_shape, in_channels = pick_dataset(validated_request.dataset, batch_size = validated_request.batch_size, normalize=True, size = None, num_workers=validated_request.num_workers)
            logger.info("Dataset successfully loaded.")
            print("shape check control:", validated_request.dataset)

            model = VanillaVAE( in_shape,in_channels, validated_request)
            # train model
            model.train_model(train_loader, validated_request.n_epochs)
            logger.info("Vanilla SGM training completed.")

            return JSONResponse(status_code=200, content={"status": "Success", "message": "Vanilla SGM training completed successfully."})

        elif request["model_name"] == "Hierarchical VAE":
            try:
                validated_request = HierarchicalVAERequestTrain(**request)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid parameters for NCSNv2: {e}")

            logger.info(f"Loading dataset for NCSNv2: {validated_request.dataset}")
            logger.info(f"Request details: {validated_request}")
            dataloader, img_size, channels = pick_dataset(validated_request.dataset, size=None, batch_size=validated_request.batch_size, num_workers=validated_request.num_workers)
            logger.info("Dataset successfully loaded.")

            model = HierarchicalVAE(validated_request.latent_dim, (img_size, img_size), channels)
            model.train_model(dataloader, validated_request)
            logger.info("HierarchicalVAE training is starting.")


            return JSONResponse(status_code=200, content={"status": "Success", "message": "Hierarchical VAE training completed successfully."})
        elif request["model_name"] == "Conditional VAE":
            try:
                validated_request = ConditionalVAERequestTrain(**request)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid parameters for NCSNv2: {e}")

            logger.info(f"Loading dataset for NCSNv2: {validated_request.dataset}")
            logger.info(f"Request details: {validated_request}")
            dataloader, img_size, channels = pick_dataset(validated_request.dataset, size=None, batch_size=validated_request.batch_size, num_workers=validated_request.num_workers)
            logger.info("Dataset successfully loaded.")
            model = ConditionalVAE(input_shape=img_size, input_channels=channels, args=validated_request)
            
            model.train_model(dataloader, validated_request.n_epochs)

            logger.info("ConditionalVAE training is starting.")

            return JSONResponse(status_code=200, content={"status": "Success", "message": "ConditionalVAE training completed successfully."})
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model: {request['model_name']}. Supported models: Vanilla SGM, ConditionalVAE."
            )

    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during model training: {str(e)}"
        )
@app.post("/vanillavae_sample")
async def sample_model(request: dict):
    try:
        
        if request["model_name"] == "Vanilla VAE":
            try:
                validated_request = VanillaVAERequestSample(**request)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid parameters for Vanilla SGM: {e}")

            logger.info(f"Loading dataset for Vanilla VAE: {validated_request.dataset}")
            _, in_shape, in_channels = pick_dataset(
                validated_request.dataset,
                'val',
                validated_request.batch_size,
                normalize=True,
                size=None
            )
            model = VanillaVAE(input_shape=in_shape, input_channels=in_channels,args=validated_request)

            checkpoint = f"/home/a30/Desktop/zoo/models/VanillaVAE/VanVAE_{validated_request.dataset}.pt"
            model.load_state_dict(torch.load(checkpoint))
            

            logger.info("Checkpoint successfully loaded.")

            # Sampling process
            logger.info(f"Generating {validated_request.num_samples} samples.")
            print("check control flag 1:")
            base64_image = model.sample(title="Sample", train = False)

            if not base64_image:
                raise ValueError("Sampling process failed. No Base64 image generated.")

            return {
                "status": "success",
                "message": "Samples generated successfully!",
                "base64_image": base64_image
            }

        elif request["model_name"] == "Hierarchical VAE":

            try:
                validated_request = HierarchicalVAERequestTrain(**request)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid parameters for NCSNv2: {e}")
            checkpoint = f"/home/a30/Desktop/zoo/models/HierarchicalVAE/HVAE_{validated_request.dataset}.pt"
            print("check control flag 0:", checkpoint)
            dataloader, img_size, channels = pick_dataset(validated_request.dataset, size=None, batch_size=validated_request.batch_size, num_workers=validated_request.num_workers)
            print("check control flag 1:")
            model = HierarchicalVAE(validated_request.latent_dim, (None, None), channels)
            print("check control flag 2:")
            if checkpoint is not None:
                model.load_state_dict(torch.load(checkpoint))
            print("check control flag 3:")
            base64_image = model.sample(validated_request)

            if not base64_image:
                raise ValueError("Sampling process failed. No Base64 image generated.")

            return {
                "status": "success",
                "message": "Samples generated successfully!",
                "base64_image": base64_image
            }
        elif request["model_name"] == "Conditional VAE":
            try:
                validated_request = ConditionalVAERequestTrain(**request)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid parameters for Vanilla SGM: {e}")
            checkpoint = f"/home/a30/Desktop/zoo/models/ConditionalVAE/CondVAE_{validated_request.dataset}.pt"
            _, in_shape, in_channels = pick_dataset(validated_request.dataset, batch_size = validated_request.batch_size, normalize=True, size=None)
            model = ConditionalVAE(input_shape=in_shape, input_channels=in_channels, args=validated_request)
            model.load_state_dict(torch.load(checkpoint))
    

            logger.info("Checkpoint successfully loaded.")

            # Sampling process
            logger.info(f"Generating {validated_request.num_samples} samples.")
            print("check control flag 1:")
            base64_image = model.sample(title="Sample", train = False)

            if not base64_image:
                raise ValueError("Sampling process failed. No Base64 image generated.")

            return {
                "status": "success",
                "message": "Samples generated successfully!",
                "base64_image": base64_image
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model: {request['model_name']}. Supported models: Vanilla SGM, NCSNv2."
            )

    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during sampling: {str(e)}"
        )

@app.post("/DDPMs_train")
async def train_model(request: dict):  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Selected device: {device}")

    try:
        # Convert JSON content into the appropriate model based on `model_name`
        if request["model_name"] == "Conditional DDPM":
            try:
                validated_request = CondDDPMRequestTrain(**request)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid parameters for Conditional DDPM: {e}")

            logger.info(f"Loading dataset for Conditional DDPM: {validated_request.dataset}")

            train_dataloader, input_size, channels = pick_dataset(validated_request.dataset, 'train', validated_request.batch_size, normalize=True, size=None, num_workers=validated_request.num_workers)
            logger.info("Dataset successfully loaded.")
            print("shape check control:", validated_request.dataset)

            model = ConditionalDDPM(in_channels=channels, input_size=input_size, args=validated_request)
            
            # train model
            model.train_model(train_dataloader)
            logger.info("Conditional DDPM training completed.")

            return JSONResponse(status_code=200, content={"status": "Success", "message": "Conditional DDPM training completed successfully."})

        elif request["model_name"] == "DAE":
            try:
                validated_request = DiffAERequestTrain(**request)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid parameters for DiffAE: {e}")

            logger.info(f"Loading dataset for DiffAE: {validated_request.dataset}")
            logger.info(f"Request details: {validated_request}")
            train_dataloader, input_size, channels = pick_dataset(validated_request.dataset, 'train', validated_request.batch_size, normalize=True, size=None, num_workers=validated_request.num_workers)

            logger.info("Dataset successfully loaded.")
            model = DiffAE(validated_request, channels)
            model.train_model(train_dataloader, train_dataloader)
            logger.info("DiffAE training is starting.")

            return JSONResponse(status_code=200, content={"status": "Success", "message": "DiffAE training completed successfully."})
        elif request["model_name"] == "Vanilla DDPM":
            try:
                validated_request = VanDDPMTrain(**request)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid parameters for vanddpm: {e}")

            logger.info(f"Loading dataset for vanddpm: {validated_request.dataset}")
            logger.info(f"Request details: {validated_request}")
            dataloader, input_size, channels = pick_dataset(validated_request.dataset, 'train', validated_request.batch_size, normalize=True, size=None, num_workers=validated_request.num_workers)
            model = VanillaDDPM(validated_request, channels=channels, image_size=input_size)
            

            # train model
            model.train_model(dataloader)

            logger.info("vanddpm training is starting.")

            return JSONResponse(status_code=200, content={"status": "Success", "message": "vanddpm training completed successfully."})
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model: {request['model_name']}. Supported models: Vanilla SGM, ConditionalVAE."
            )

    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during model training: {str(e)}"
        )
    
@app.post("/DDPMs_sample")
async def sample_model(request: dict):
    try:
        # Pydantic doğrulama ve model seçimi
        if request["model_name"] == "Conditional DDPM":
            try:
                validated_request = CondDDPMRequestTrain(**request)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid parameters for Vanilla SGM: {e}")

            logger.info(f"Loading dataset for cond ddpm: {validated_request.dataset}")
            _, input_size, channels = pick_dataset(validated_request.dataset, 'train', validated_request.batch_size, normalize=True, size=None)
            model = ConditionalDDPM(in_channels=channels, input_size=input_size, args=validated_request)
            
            checkpoint = f"/home/a30/Desktop/zoo/models/ConditionalDDPM/CondDDPM_{validated_request.dataset}.pt"
            print("check control flag 0:", checkpoint)
            model.denoising_model.load_state_dict(torch.load(checkpoint))
            
            logger.info("Checkpoint successfully loaded.")

            # Sampling process
            logger.info(f"Generating {validated_request.num_samples} samples.")
           
            base64_image = model.sample(guide_w=validated_request.guide_w)

            if not base64_image:
                raise ValueError("Sampling process failed. No Base64 image generated.")

            return {
                "status": "success",
                "message": "Samples generated successfully!",
                "base64_image": base64_image
            }

        elif request["model_name"] == "Vanilla DDPM":
            try:
                validated_request = VanDDPMTrain(**request)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid parameters for vam ddpm: {e}")
            checkpoint = f"/home/a30/Desktop/zoo/models/VanillaDDPM/VanDDPM_{validated_request.dataset}.pt"
            print("check control flag 0:", checkpoint)
            _, input_size, channels = pick_dataset(validated_request.dataset, 'val', validated_request.batch_size, normalize=True, size=None)
            model = VanillaDDPM(validated_request, channels=channels, image_size=input_size)
            model.denoising_model.load_state_dict(torch.load(checkpoint))
            
            print("check control flag 3:")
            base64_image = model.sample(validated_request.num_samples)

            if not base64_image:
                raise ValueError("Sampling process failed. No Base64 image generated.")

            return {
                "status": "success",
                "message": "Samples generated successfully!",
                "base64_image": base64_image
            }
        elif request["model_name"] == "Conditional VAE":
            try:
                validated_request = ConditionalVAERequestTrain(**request)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid parameters for Vanilla SGM: {e}")
            checkpoint = f"/home/a30/Desktop/zoo/models/ConditionalVAE/CondVAE_{validated_request.dataset}.pt"
            _, in_shape, in_channels = pick_dataset(validated_request.dataset, batch_size = validated_request.batch_size, normalize=True, size=None)
            model = ConditionalVAE(input_shape=in_shape, input_channels=in_channels, args=validated_request)
            model.load_state_dict(torch.load(checkpoint))
    

            logger.info("Checkpoint successfully loaded.")

            # Sampling process
            logger.info(f"Generating {validated_request.num_samples} samples.")
            print("check control flag 1:")
            base64_image = model.sample(title="Sample", train = False)

            if not base64_image:
                raise ValueError("Sampling process failed. No Base64 image generated.")

            return {
                "status": "success",
                "message": "Samples generated successfully!",
                "base64_image": base64_image
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model: {request['model_name']}. Supported models: Vanilla SGM, NCSNv2."
            )

    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during sampling: {str(e)}"
        )
@app.post("/Autoregressive_train")
async def train_model(request: dict):  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Selected device: {device}")

    try:
        # Convert JSON content into the appropriate model based on `model_name`
        if request["model_name"] == "PixelCNN":
            try:
                validated_request = PixelCNNTrain(**request)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid parameters for PixelCNN: {e}")

            logger.info(f"Loading dataset for PixelCNN: {validated_request.dataset}")

            dataloader, img_size, channels = pick_dataset(validated_request.dataset, normalize=False, batch_size=validated_request.batch_size, size=None, num_workers=validated_request.num_workers)
            logger.info("Dataset successfully loaded.")
            print("shape check control:", validated_request.dataset)

            model = PixelCNN(channels, validated_request.hidden_channels)
            model.train_model(dataloader, validated_request, img_size)
            logger.info("Pixel CNN training completed.")

            return JSONResponse(status_code=200, content={"status": "Success", "message": "Pixel CNN training completed successfully."})

        elif request["model_name"] == "VQ-GAN + Transformer":
            try:
                validated_request = VQGANTransformerTrain(**request)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid parameters for DiffAE: {e}")

            logger.info(f"Loading dataset for vqgan: {validated_request.dataset}")
            logger.info(f"Request details: {validated_request}")
            train_loader_a, input_size, channels = pick_dataset(validated_request.dataset, 'train', validated_request.batch_size, normalize=False, size=None, num_workers=validated_request.num_workers)
            train_loader_b, _, _ = pick_dataset(validated_request.dataset, 'train', validated_request.batch_size//4, normalize=False, size=None, num_workers=validated_request.num_workers)
            print("shape check control:", validated_request.n_epochs)
            logger.info("Dataset successfully loaded.")
            model = VQGANTransformer(validated_request, channels=channels, img_size=input_size)
            model.train_model(validated_request, train_loader_a, train_loader_b)
            logger.info("vqgan training is starting.")

            return JSONResponse(status_code=200, content={"status": "Success", "message": "VQGAN training completed successfully."})
        elif request["model_name"] == "VQ-VAE + Transformer":
            try:
                validated_request = VQVAETransformerTrain(**request)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid parameters for VQ-VAE: {e}")

            logger.info(f"Loading dataset for vanddpm: {validated_request.dataset}")
            logger.info(f"Request details: {validated_request}")
            train_loader_a, input_size, channels = pick_dataset(validated_request.dataset, 'train', validated_request.batch_size, normalize=False, size=None, num_workers=validated_request.num_workers)
            train_loader_b, _, _ = pick_dataset(validated_request.dataset, 'train', validated_request.batch_size//4, normalize=False, size=None, num_workers=validated_request.num_workers)
            model = VQVAETransformer(validated_request, channels=channels, img_size=input_size)
            model.train_model(validated_request, train_loader_a, train_loader_b)

            logger.info("VQVAE training is starting.")

            return JSONResponse(status_code=200, content={"status": "Success", "message": "VQVAE training completed successfully."})
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model: {request['model_name']}. Supported models: Vanilla SGM, ConditionalVAE."
            )

    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during model training: {str(e)}"
        )
    
@app.post("/Autoregressive_sample")
async def sample_model(request: dict):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      
        if request["model_name"] == "PixelCNN":
            try:
                validated_request = PixelCNNTrain(**request)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid parameters for PixelCNN: {e}")

            logger.info(f"Loading dataset for PixelCNN: {validated_request.dataset}")
            _, img_size, channels = pick_dataset(validated_request.dataset, normalize=False, batch_size=validated_request.batch_size, size=None)
            model = PixelCNN(channels, validated_request.hidden_channels)

            checkpoint = f"/home/a30/Desktop/zoo/models/PixelCNN/PixelCNN_{validated_request.dataset}.pt"
            model.load_state_dict(torch.load(checkpoint))
            logger.info("Checkpoint successfully loaded.")

            base64_image = model.sample((16,channels,img_size,img_size), train=False)
        
            if not base64_image:
                raise ValueError("Sampling process failed. No Base64 image generated.")

            return {
                "status": "success",
                "message": "Samples generated successfully!",
                "base64_image": base64_image
            }

        elif request["model_name"] == "VQ-GAN + Transformer":
            try:
                validated_request = VQGANTransformerTrain(**request)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid parameters for DiffAE: {e}")

            logger.info(f"Loading dataset for vqgan: {validated_request.dataset}")
            logger.info(f"Request details: {validated_request}")

           
            _, input_size, channels = pick_dataset(validated_request.dataset, 'train', 1, normalize=False, size=None)
            model = VQGANTransformer(validated_request, channels=channels, img_size=input_size)
            checkpoint = f"/home/a30/Desktop/zoo/models/VQGAN_Transformer/VQGAN_{validated_request.dataset}.pt"
            checkpoint_t = f"/home/a30/Desktop/zoo/models/VQGAN_Transformer/Transformer_{validated_request.dataset}.pt"
            model.load_checkpoint(checkpoint, checkpoint_t)
            model.eval()
            base64_image = model.sample(16, train=False)
            
            print("check control flag 3:")

            if not base64_image:
                raise ValueError("Sampling process failed. No Base64 image generated.")

            return {
                "status": "success",
                "message": "Samples generated successfully!",
                "base64_image": base64_image
            }
        elif request["model_name"] == "VQ-VAE + Transformer":
            try:
                validated_request = VQVAETransformerTrain(**request)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid parameters for VQ-VAE: {e}")

            logger.info(f"Loading dataset for vanddpm: {validated_request.dataset}")
            logger.info(f"Request details: {validated_request}")

            checkpoint = f"/home/a30/Desktop/zoo/models/VQVAE_Transformer/VQVAE_{validated_request.dataset}.pt"
            checkpoint_t = f"/home/a30/Desktop/zoo/models/VQVAE_Transformer/Transformer_{validated_request.dataset}.pt"
            _, input_size, channels = pick_dataset(validated_request.dataset, 'train', 1, normalize=False, size=None)
            model = VQVAETransformer(validated_request, channels=channels, img_size=input_size)
            model.load_checkpoint(checkpoint, checkpoint_t)
            

            logger.info("Checkpoint successfully loaded.")

            # Sampling process
            logger.info(f"Generating {validated_request.num_samples} samples.")
            print("check control flag 1:")
            base64_image = model.sample(16, train=False)

            if not base64_image:
                raise ValueError("Sampling process failed. No Base64 image generated.")

            return {
                "status": "success",
                "message": "Samples generated successfully!",
                "base64_image": base64_image
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model: {request['model_name']}. Supported models: Vanilla SGM, NCSNv2."
            )

    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during sampling: {str(e)}"
        )
    
@app.post("/GAN_train")
async def train_model(request: dict):  # Raw JSON data is received
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Selected device: {device}")

    try:
        # Convert JSON content into the appropriate model based on `model_name`
        if request["model_name"] == "Adversarial VAE":
            try:
                validated_request = AdversarialVAETrain(**request)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid parameters for Adversarial VAE: {e}")

            logger.info(f"Loading dataset for Adversarial VAE: {validated_request.dataset}")

            train_loader, input_size, channels = pick_dataset(dataset_name=validated_request.dataset, batch_size=validated_request.batch_size, normalize=True, num_workers=validated_request.num_workers, mode='train', size=validated_request.size)
            model = AdversarialVAE(input_shape = input_size, input_channels=channels, args=validated_request)
            model.train_model(train_loader)
            logger.info("Adversarial VAE training completed.")

            return JSONResponse(status_code=200, content={"status": "Success", "message": "Adversarial VAE training completed successfully."})

        elif request["model_name"] == "Vanilla GAN":
            try:
                validated_request = VanillaGANTrain(**request)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid parameters for Vanilla GAN: {e}")

            logger.info(f"Loading dataset for Vanilla GAN: {validated_request.dataset}")
            logger.info(f"Request details: {validated_request}")
            train_dataloader, input_size, channels = pick_dataset(dataset_name = validated_request.dataset, batch_size=validated_request.batch_size, normalize = True, size = None, num_workers=validated_request.num_workers)

            logger.info("Dataset successfully loaded.")
            model = VanillaGAN(validated_request, channels, input_size)
            model.train_model(train_dataloader)
            logger.info("Vanilla GAN training is starting.")

            return JSONResponse(status_code=200, content={"status": "Success", "message": "Vanilla GAN training completed successfully."})
        elif request["model_name"] == "Conditional GAN":
            try:
                validated_request = CondGANTrain(**request)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid parameters for Conditional: {e}")

            logger.info(f"Loading dataset for Conditional: {validated_request.dataset}")
            logger.info(f"Request details: {validated_request}")
            train_dataloader, input_size, channels = pick_dataset(dataset_name = validated_request.dataset, batch_size=validated_request.batch_size, normalize = True, size = None, num_workers=validated_request.num_workers)
            model = ConditionalGAN(img_size=input_size, channels=channels, args=validated_request)
            model.train_model(train_dataloader)
            logger.info("Conditional training is starting.")

            return JSONResponse(status_code=200, content={"status": "Success", "message": "Conditional training completed successfully."})
        elif request["model_name"] == "CycleGAN":
            try:
                validated_request = CycleGANTrain(**request)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid parameters for Conditional: {e}")

            logger.info(f"Loading dataset for Conditional: {validated_request.dataset}")
            logger.info(f"Request details: {validated_request}")
            train_dataloader_A = get_horse2zebra_dataloader(data_raw_dir, validated_request.dataset, validated_request.batch_size, True, 'A', validated_request.input_size)
            train_dataloader_B = get_horse2zebra_dataloader(data_raw_dir, validated_request.dataset, validated_request.batch_size, True, 'B', validated_request.input_size)
            test_dataloader_A = get_horse2zebra_dataloader(data_raw_dir, validated_request.dataset, validated_request.batch_size, False, 'A', validated_request.input_size)
            test_dataloader_B = get_horse2zebra_dataloader(data_raw_dir, validated_request.dataset, validated_request.batch_size, False, 'B', validated_request.input_size)
            model = CycleGAN(validated_request.in_channels, validated_request.out_channels, validated_request)
            model.train_model(train_dataloader_A, train_dataloader_B, test_dataloader_A, test_dataloader_B)
            logger.info("Conditional training is starting.")

            return JSONResponse(status_code=200, content={"status": "Success", "message": "Conditional training completed successfully."})
        elif request["model_name"] == "Prescribed GAN":
            try:
                validated_request = PresGANTrain(**request)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid parameters for Prescribed GAN: {e}")

            logger.info(f"Loading dataset for Conditional: {validated_request.dataset}")
            logger.info(f"Request details: {validated_request}")
            train_dataloader, input_size, channels = pick_dataset(dataset_name = validated_request.dataset, batch_size=validated_request.batch_size, normalize = True, size=None, num_workers=validated_request.num_workers)
            model = PresGAN(imgSize=input_size, channels=channels, args=validated_request)
            model.train_model(train_dataloader)
            logger.info("Prescribed GAN training is starting.")

            return JSONResponse(status_code=200, content={"status": "Success", "message": "Prescribed GAN training completed successfully."})
        elif request["model_name"] == "Wasserstein GAN with Gradient Penalty":
            try:
                validated_request = WassersteinGANTrain(**request)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid parameters for Wasserstein GAN: {e}")

            logger.info(f"Loading dataset for Wasserstein GAN: {validated_request.dataset}")
            logger.info(f"Request details: {validated_request}")
            train_loader, input_size, channels = pick_dataset(dataset_name=validated_request.dataset, batch_size=validated_request.batch_size, normalize=False, size=None, num_workers=validated_request.num_workers)
            model = WGAN(args=validated_request, imgSize=input_size, channels=channels)
            model.train_model(train_loader)
            logger.info("Wasserstein GAN training is starting.")

            return JSONResponse(status_code=200, content={"status": "Success", "message": "Wasserstein GAN training completed successfully."})
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model: {request['model_name']}. Supported models: Vanilla SGM, ConditionalVAE."
            )

    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during model training: {str(e)}"
        )
    
@app.post("/GAN_sample")
async def sample_model(request: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Selected device: {device}")

    try:
        # Convert JSON content into the appropriate model based on `model_name`
        if request["model_name"] == "Adversarial VAE":
            try:
                validated_request = AdversarialVAETrain(**request)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid parameters for Conditional DDPM: {e}")

            logger.info(f"Loading dataset for Conditional DDPM: {validated_request.dataset}")

            _, input_size, channels = pick_dataset(dataset_name=validated_request.dataset, batch_size=validated_request.batch_size, normalize=True, mode='val', size=None)

            model = AdversarialVAE(input_shape = input_size, input_channels=channels, args=validated_request)
            
            checkpoint = f"/home/a30/Desktop/zoo/models/AdversarialVAE/AdvVAE_{validated_request.dataset}.pt"
            print("check control flag 0:", checkpoint)
            model.load_state_dict(torch.load(checkpoint))
            
            logger.info("Checkpoint successfully loaded.")

            logger.info(f"Generating {validated_request.num_samples} samples.")
            print("check control flag 1:")
            base64_image = model.create_grid()
        
    
            if not base64_image:
                raise ValueError("Sampling process failed. No Base64 image generated.")

            return {
                "status": "success",
                "message": "Samples generated successfully!",
                "base64_image": base64_image
            }

        elif request["model_name"] == "Vanilla GAN":
            try:
                validated_request = VanillaGANTrain(**request)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid parameters for Vanilla GAN: {e}")

            logger.info(f"Loading dataset for Vanilla GAN: {validated_request.dataset}")
            logger.info(f"Request details: {validated_request}")

            checkpoint_path = f"/home/a30/Desktop/zoo/models/VanillaGAN/VanDisc_{validated_request.dataset}.pt"
            print("check control flag 0:", checkpoint_path)

            _, input_size, channels = pick_dataset(dataset_name = validated_request.dataset, batch_size=1, normalize=True, size=None)
            model = Generator(latent_dim = validated_request.latent_dim, channels = channels, d=validated_request.d, imgSize=input_size).to(device)

            
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

            new_state_dict = {key.replace("generator.", ""): value for key, value in checkpoint.items()}

            model.load_state_dict(new_state_dict, strict=False)
            model.eval()

            print("check control flag 3:")
            base64_image = model.sample(n_samples = validated_request.n_samples, device = device)

            if not base64_image:
                raise ValueError("Sampling process failed. No Base64 image generated.")

            return {
                "status": "success",
                "message": "Samples generated successfully!",
                "base64_image": base64_image
            }
        elif request["model_name"] == "Conditional GAN":
            try:
                validated_request = CondGANTrain(**request)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid parameters for Conditional: {e}")

            checkpoint = f"/home/a30/Desktop/zoo/models/ConditionalGAN/CondGAN_{validated_request.dataset}.pt"
            
            _, input_size, channels = pick_dataset(
                dataset_name=validated_request.dataset,
                batch_size=1,
                normalize=True,
                size=None
            )

          
            model = Generator(
                latent_dim=validated_request.latent_dim,
                channels=channels,
                d=validated_request.d
            ).to(device)

            try:
                checkpoint_data = torch.load(checkpoint, map_location="cpu")

          
                new_state_dict = {}
                for key, value in checkpoint_data.items():
                    new_key = key.replace("generator.", "")  
                    new_state_dict[new_key] = value

                model.load_state_dict(new_state_dict, strict=False)
                model.eval()
                logger.info("Checkpoint successfully loaded.")

            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}")
                raise HTTPException(status_code=500, detail=f"Model checkpoint loading failed: {e}")

            # Sampling süreci
            logger.info(f"Generating {validated_request.num_samples} samples.")
            print("check control flag 1:")

            try:
                base64_image = model.sample(n_samples=validated_request.num_samples, device=device)

                if not base64_image:
                    raise ValueError("Sampling process failed. No Base64 image generated.")

                return {
                    "status": "success",
                    "message": "Samples generated successfully!",
                    "base64_image": base64_image
                }

            except Exception as e:
                logger.error(f"Error during sampling: {e}")
                raise HTTPException(status_code=500, detail=f"Image generation failed: {e}")
        elif request["model_name"] == "Prescribed GAN":
            try:
                validated_request = PresGANTrain(**request)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid parameters for Prescribed GAN: {e}")
            checkpoint = f"/home/a30/Desktop/zoo/models/PrescribedGAN/PresGAN_{validated_request.dataset}.pt"
            logger.info(f"Loading dataset for Conditional: {validated_request.dataset}")
            logger.info(f"Request details: {validated_request}")
            _, input_size, channels = pick_dataset(dataset_name = validated_request.dataset, batch_size=validated_request.batch_size, normalize = True, size = None)
            model = PresGAN(imgSize=input_size, channels=channels, args=validated_request)
            model.load_checkpoints(generator_checkpoint=validated_request.checkpoint, discriminator_checkpoint=validated_request.discriminator_checkpoint, sigma_checkpoint=validated_request.sigma_checkpoint)
            base64_image = model.sample(num_samples=validated_request.num_gen_images)
            if not base64_image:
                raise ValueError("Sampling process failed. No Base64 image generated.")

            return {
                "status": "success",
                "message": "Samples generated successfully!",
                "base64_image": base64_image
            }
        elif request["model_name"] == "Wasserstein GAN with Gradient Penalty":
            try:
                validated_request = WassersteinGANTrain(**request)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid parameters for Wasserstein GAN: {e}")

            logger.info(f"Loading dataset for Wasserstein GAN: {validated_request.dataset}")
            logger.info(f"Request details: {validated_request}")

            checkpoint = f"/home/a30/Desktop/zoo/models/WassersteinGAN/WGAN_{validated_request.dataset}.pt"

            _, input_size, channels = pick_dataset(dataset_name=validated_request.dataset, batch_size=1, normalize=False, size=None)
            model = Generator(latent_dim=validated_request.latent_dim, channels=channels, d=validated_request.d, imgSize=input_size).to(device)
            model.load_state_dict(torch.load(checkpoint))
              
            base64_image = model.sample(n_samples=validated_request.n_samples, device=device)  
            if not base64_image:
                raise ValueError("Sampling process failed. No Base64 image generated.")

            return {
                "status": "success",
                "message": "Samples generated successfully!",
                "base64_image": base64_image
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model: {request['model_name']}. Supported models: Vanilla SGM, NCSNv2."
            )

    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during sampling: {str(e)}"
        )
    
@app.post("/Flow_train")
async def train_model(request: dict):  # Raw JSON data is received
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Selected device: {device}")

    try:
        # Convert JSON content into the appropriate model based on `model_name`
        if request["model_name"] == "Vanilla Flow":
            try:
                validated_request = VanillaFlowTrain(**request)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid parameters for Vanilla Flow: {e}")

            logger.info(f"Loading dataset for Vanilla Flow: {validated_request.dataset}")

            in_loader, img_size, channels = pick_dataset(validated_request.dataset, 'train', validated_request.batch_size, num_workers=validated_request.num_workers)
            logger.info("Vanilla Flow training completed.")
            model = VanillaFlow(img_size, channels, validated_request)
            model.train_model(in_loader, validated_request)
            return JSONResponse(status_code=200, content={"status": "Success", "message": "Vanilla Flow training completed successfully."})

        elif request["model_name"] == "RealNVP":
            try:
                validated_request = RealNVPTrain(**request)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid parameters for RealNVP: {e}")

            logger.info(f"Loading dataset for RealNVP: {validated_request.dataset}")
            logger.info(f"Request details: {validated_request}")
            dataloader, img_size, channels = pick_dataset(validated_request.dataset, batch_size=validated_request.batch_size, normalize = False, size=None, num_workers=validated_request.num_workers)
            model = RealNVP(img_size=img_size, in_channels=channels, args=validated_request)
            logger.info("Dataset successfully loaded.")
            model.train_model(dataloader, validated_request)
            logger.info("RealNVP training is starting.")

            return JSONResponse(status_code=200, content={"status": "Success", "message": "RealNVP training completed successfully."})
        elif request["model_name"] == "Glow":
            try:
                validated_request = GlowTrain(**request)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid parameters for Glow: {e}")

            logger.info(f"Loading dataset for Glow: {validated_request.dataset}")
            logger.info(f"Request details: {validated_request}")
            train_loader, input_shape, channels = pick_dataset(validated_request.dataset, batch_size=validated_request.batch_size, normalize=False, size=None, num_workers=validated_request.num_workers)
            model = Glow(image_shape        =   (input_shape,input_shape,channels), hidden_channels    =   validated_request.hidden_channels, args=validated_request)
            model.train_model(train_loader, validated_request)
            logger.info("Glow training is starting.")

            return JSONResponse(status_code=200, content={"status": "Success", "message": "Glow training completed successfully."})
        elif request["model_name"] == "Flow++":
            try:
                validated_request = FlowPPTrain(**request)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid parameters for Flow++: {e}")

            logger.info(f"Loading dataset for Flow++: {validated_request.dataset}")
            logger.info(f"Request details: {validated_request}")
            train_loader, input_size, channels = pick_dataset(validated_request.dataset, 'train', validated_request.batch_size, normalize=False, size=None, num_workers=validated_request.num_workers)
            
            model = FlowPlusPlus(validated_request, channels=channels, img_size=input_size)
            model.train_model(validated_request, train_loader)
            logger.info("Flow++ training is starting.")

            return JSONResponse(status_code=200, content={"status": "Success", "message": "Flow++ training completed successfully."})
        elif request["model_name"] == "Flow Matching":
            try:
                validated_request = FlowMatchingTrain(**request)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid parameters for Flow: {e}")

            logger.info(f"Loading dataset for Flow Matching: {validated_request.dataset}")
            logger.info(f"Request details: {validated_request}")
            train_loader, input_size, channels = pick_dataset(validated_request.dataset, batch_size = validated_request.batch_size, normalize=True, num_workers=validated_request.num_workers)           
            logger.info("Flow training is starting.")
            model = FlowMatching(validated_request, input_size, channels)
            model.train_model(train_loader)
            return JSONResponse(status_code=200, content={"status": "Success", "message": "Flow training completed successfully."})
        elif request["model_name"] == "Conditional Flow Matching":
            try:
                validated_request = CondFlowMatchingTrain(**request)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid parameters for Conditional Flow: {e}")

            logger.info(f"Loading dataset for Conditional Flow: {validated_request.dataset}")
            logger.info(f"Request details: {validated_request}")
            train_loader, input_size, channels = pick_dataset(validated_request.dataset, batch_size = validated_request.batch_size, normalize=True, num_workers=validated_request.num_workers)
            model = CondFlowMatching(validated_request, input_size, channels)
            model.train_model(train_loader)
            logger.info("Conditional Flow training is starting.")

            return JSONResponse(status_code=200, content={"status": "Success", "message": "Conditional Flow training completed successfully."})
        elif request["model_name"] == "Rectified Flows":
            try:
                validated_request = RectifiedFlowsTrain(**request)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid parameters for Rectified Flow: {e}")

            logger.info(f"Loading dataset for Rectified Flow: {validated_request.dataset}")
            logger.info(f"Request details: {validated_request}")
            train_loader, input_size, channels = pick_dataset(validated_request.dataset, batch_size = validated_request.batch_size, normalize=True, num_workers=validated_request.num_workers)
            model = RF(validated_request, input_size, channels)
            model.train_model(train_loader)
            logger.info("Rectified Flow training is starting.")

            return JSONResponse(status_code=200, content={"status": "Success", "message": "Rectified Flow training completed successfully."})
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model: {request['model_name']}. Supported models: Vanilla SGM, ConditionalVAE."
            )

    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during model training: {str(e)}"
        )
    

@app.post("/Flow_sample")
async def sample_model(request: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Selected device: {device}")

    try:
        # Convert JSON content into the appropriate model based on `model_name`
        if request["model_name"] == "Vanilla Flow":
            try:
                validated_request = VanillaFlowTrain(**request)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid parameters for Vanilla Flow: {e}")

            logger.info(f"Loading dataset for Vanilla Flow: {validated_request.dataset}")

            _, img_size, channels = pick_dataset(validated_request.dataset, 'val', validated_request.batch_size)

            model = VanillaFlow(img_size, channels, validated_request)
            
            checkpoint = f"/home/a30/Desktop/zoo/models/VanillaFlow/VanFlow_{validated_request.dataset}.pt"
            print("check control flag 0:", checkpoint)
            model.flows.load_state_dict(torch.load(checkpoint))
            
            logger.info("Checkpoint successfully loaded.")

            # Sampling process
            logger.info(f"Generating {validated_request.num_samples} samples.")
            print("check control flag 1:")
            base64_image = model.sample(train=False)
            if not base64_image:
                raise ValueError("Sampling process failed. No Base64 image generated.")

            return {
                "status": "success",
                "message": "Samples generated successfully!",
                "base64_image": base64_image
            }

        elif request["model_name"] == "RealNVP":
            try:
                validated_request = RealNVPTrain(**request)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid parameters for RealNVP: {e}")

            logger.info(f"Loading dataset for RealNVP: {validated_request.dataset}")
            logger.info(f"Request details: {validated_request}")

            checkpoint = f"/home/a30/Desktop/zoo/models/RealNVP/RealNVP_{validated_request.dataset}.pt"
            print("check control flag 0:", checkpoint)
            _, img_size, channels = pick_dataset(validated_request.dataset, batch_size=1, normalize = False, size=None)
            model = RealNVP(img_size=img_size, in_channels=channels, args=validated_request)
            model.load_state_dict(torch.load(checkpoint))

            
            print("check control flag 3:")
            base64_image = model.sample(16, train=False)
    
            if not base64_image:
                raise ValueError("Sampling process failed. No Base64 image generated.")

            return {
                "status": "success",
                "message": "Samples generated successfully!",
                "base64_image": base64_image
            }
        elif request["model_name"] == "Glow":
            try:
                validated_request = GlowTrain(**request)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid parameters for Glow: {e}")

            logger.info(f"Loading dataset for Glow: {validated_request.dataset}")
            logger.info(f"Request details: {validated_request}")

            
            validated_request.checkpoint = f"/home/a30/Desktop/zoo/models/Glow/Glow_{validated_request.dataset}.pt"
            _, input_shape, channels = pick_dataset(validated_request.dataset, batch_size=validated_request.batch_size, normalize=False, size=None, num_workers=0)
            model = Glow(image_shape        =   (input_shape,input_shape,channels), hidden_channels    =   validated_request.hidden_channels, args=validated_request)
            model.load_checkpoint(validated_request)
    

            logger.info("Checkpoint successfully loaded.")

            # Sampling process
            logger.info(f"Generating {validated_request.num_samples} samples.")
            print("check control flag 1:")
            base64_image = model.sample(train=False)
    
            if not base64_image:
                raise ValueError("Sampling process failed. No Base64 image generated.")

            return {
                "status": "success",
                "message": "Samples generated successfully!",
                "base64_image": base64_image
            }
        elif request["model_name"] == "Flow Matching":
            try:
                validated_request = FlowMatchingTrain(**request)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid parameters for Flow: {e}")

            logger.info(f"Loading dataset for Flow Matching: {validated_request.dataset}")
            logger.info(f"Request details: {validated_request}")
            checkpoint = f"/home/a30/Desktop/zoo/models/FlowMatching/FM_{validated_request.dataset}.pt"
            logger.info(f"Loading dataset for Conditional: {validated_request.dataset}")
            logger.info(f"Request details: {validated_request}")
            _, input_size, channels = pick_dataset(validated_request.dataset, batch_size = 1, normalize=True)
            model = FlowMatching(validated_request, input_size, channels)
            model.load_checkpoint(checkpoint)
            base64_image = model.sample(validated_request.num_samples, train=False)
            if not base64_image:
                raise ValueError("Sampling process failed. No Base64 image generated.")
        
            return {
                "status": "success",
                "message": "Samples generated successfully!",
                "base64_image": base64_image
            }
        elif request["model_name"] == "Conditional Flow Matching":
            try:
                validated_request = CondFlowMatchingTrain(**request)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid parameters for Conditional Flow: {e}")

            logger.info(f"Loading dataset for Conditional Flow: {validated_request.dataset}")
            logger.info(f"Request details: {validated_request}")

            checkpoint = f"/home/a30/Desktop/zoo/models/CondFlowMatching/CondFM_{validated_request.dataset}.pt"

            _, input_size, channels = pick_dataset(validated_request.dataset, batch_size = 1, normalize=True)
            model = CondFlowMatching(validated_request, input_size, channels)
            model.load_checkpoint(checkpoint)
              
            base64_image = model.sample(validated_request.guidance_scale, train=False) 
   
            if not base64_image:
                raise ValueError("Sampling process failed. No Base64 image generated.")

            return {
                "status": "success",
                "message": "Samples generated successfully!",
                "base64_image": base64_image
            }
        elif request["model_name"] == "Rectified Flows":
            print("Check contorl -1")
            try:
                validated_request = RectifiedFlowsTrain(**request)
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid parameters for Rectified Flow: {e}")


            checkpoint = f"/home/a30/Desktop/zoo/models/RectifiedFlows/RF_{validated_request.dataset}.pt"
            print("Check contorl -1")
            _, input_size, channels = pick_dataset(validated_request.dataset, batch_size = 1, normalize=True)
            model = RF(validated_request, input_size, channels)
            model.load_checkpoint(checkpoint)
              
            base64_image = model.sample(16)
   
            if not base64_image:
                raise ValueError("Sampling process failed. No Base64 image generated.")
        
            return {
                "status": "success",
                "message": "Samples generated successfully!",
                "base64_image": base64_image
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model: {request['model_name']}. Supported models: Vanilla SGM, NCSNv2."
            )

    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during sampling: {str(e)}"
        )  

MODEL_DIR = "/home/a30/Desktop/zoo/models"

# YAML dosyasını yükle
yaml_path = "model_directory_mapping.yaml"
with open(yaml_path, "r") as f:
    model_mapping = yaml.safe_load(f)["models"]

@app.post("/ModelDownload")
async def download_model_zip(request: dict):
    """
    Downloads the specified model's files as a ZIP archive based on the model name and dataset.

    Request body:
    {
        "Model": "PixelCNN",
        "Dataset": "mnist"
    }

    Returns:
        - ZIP file if model and dataset match is found.
        - 404 error if the model, folder, or matching dataset files are not found.
    """
    model_name = request.get("Model")
    dataset_name = request.get("Dataset")

    if not model_name or not dataset_name:
        raise HTTPException(status_code=400, detail="Both 'Model' and 'Dataset' must be provided.")

    print(f"Requested Model: {model_name}")
    print(f"Requested Dataset: {dataset_name}")

    folder_name = model_mapping.get(model_name)
    if not folder_name:
        raise HTTPException(status_code=404, detail=f"No folder mapping found for model '{model_name}'.")

    model_path = os.path.join(MODEL_DIR, folder_name)
    if not os.path.isdir(model_path):
        raise HTTPException(status_code=404, detail=f"Model folder '{folder_name}' not found.")

    matched_files = [f for f in os.listdir(model_path) if f.endswith(".pt") and dataset_name in f]
    if not matched_files:
        raise HTTPException(status_code=404, detail=f"No model files found for dataset '{dataset_name}' in model '{model_name}'.")

    with tempfile.TemporaryDirectory() as temp_dir:
        for filename in matched_files:
            shutil.copy2(os.path.join(model_path, filename), os.path.join(temp_dir, filename))
        zip_base = tempfile.mktemp()
        shutil.make_archive(zip_base, 'zip', temp_dir)
        zip_file_path = f"{zip_base}.zip"

    return FileResponse(
        zip_file_path,
        filename=f"{model_name}_{dataset_name}.zip",
        media_type="application/zip"
    )