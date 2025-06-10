from agentflow.core.instructions.image import ImageProcessingInstruction, MockImageProcessor, MockImageClassifier
from agentflow.core.instructions.base import InstructionResult
from agentflow.core.isa_manager import ISAManager, Instruction, InstructionType
import ray
import torch
import asyncio
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

async def main():
    # Initialize Ray with proper configuration
    if not ray.is_initialized():
        try:
            ray.init(
                num_cpus=2,  # Specify number of CPUs
                ignore_reinit_error=True,
                logging_level=logging.INFO,
                include_dashboard=False  # Disable dashboard to reduce overhead
            )
            logger.info("Ray initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Ray: {e}")
            return
    
    try:
        # Create ISA manager
        isa_manager = ISAManager()
        logger.info("Created ISA manager")
        
        # Create image processing instruction with mock models for testing
        try:
            image_processor = ImageProcessingInstruction.remote()
            logger.info("Created image processing instruction")
        except Exception as e:
            logger.error(f"Failed to create image processor: {e}")
            return
        
        # Register instruction
        instruction = Instruction(
            id="process_image",
            name="process_image",
            type=InstructionType.COMPUTATION,
            description="Process and analyze images",
            params={},
            parallelizable=True
        )
        isa_manager.register_instruction(instruction)
        logger.info("Registered instruction")
        
        # Test single image processing
        try:
            single_context = {"image": torch.randn(3, 224, 224)}
            logger.info("Processing single image...")
            result = ray.get(image_processor.execute.remote(single_context))
            logger.info(f"Single image result: {result}")
        except Exception as e:
            logger.error(f"Failed to process single image: {e}")
        
        # Test batch processing
        try:
            batch_context = {"images": [torch.randn(3, 224, 224) for _ in range(5)]}
            logger.info("Processing batch of images...")
            result = ray.get(image_processor.execute.remote(batch_context))
            logger.info(f"Batch processing result: {result}")
        except Exception as e:
            logger.error(f"Failed to process batch: {e}")
        
        # Test optimization
        try:
            optimization_context = {"image": torch.randn(3, 448, 448)}
            logger.info("Testing optimization...")
            result = ray.get(image_processor.execute.remote(optimization_context))
            logger.info(f"Optimization result: {result}")
        except Exception as e:
            logger.error(f"Failed to test optimization: {e}")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    
    finally:
        # Cleanup
        try:
            await isa_manager.cleanup()
            logger.info("ISA manager cleanup completed")
        except Exception as e:
            logger.error(f"Failed to cleanup ISA manager: {e}")
        
        try:
            ray.shutdown()
            logger.info("Ray shutdown completed")
        except Exception as e:
            logger.error(f"Failed to shutdown Ray: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 