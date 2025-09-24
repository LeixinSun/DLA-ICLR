# DLA-ICLR Project

## Repository Structure

```
DLA-ICLR/
├── cras/          # Contextualized Role Adherence Score
├── sail/          # Surgical Alignment of Instruction Layers
```

## CRAS - Contextualized Role Adherence Score

### Core Components
- `rubric_generator.py`: Creates scoring rubrics for specific roles (Logician, Physicist, etc.)
- `evaluate_scores.py`: Main scoring engine with cost management and concurrent processing
- `prompts/`: Template prompts for rubric generation and multi-dimensional scoring
- `configs/`: API configuration for different model providers (DeepSeek, OpenAI, etc.)

## SAIL - Surgical Alignment of Instruction Layers

### Core Components
- `src/llamafactory/`: Training framework based on LlamaFactory
- `src/train.py`: Main training entry point (disables WANDB by default)
- `llama_sail_config.yaml`: Configuration file with SAIL-specific parameters

### Training Configuration
SAIL's surgical approach uses specific configurations:
- **Base Model**
- **Reward Model** for token-level reward signals
- **Surgical Layers**: Only focal layers
- **LoRA Parameters**: Rank 8, alpha 16, dropout 0.0 
- **SAIL Alpha**: 0.5 parameter controls the balance between preference and reward learning
- **Hardware**: Multi-GPU distributed training with FSDP
- **Dataset**: FOCAL dataset

## Getting Started

### Quick Setup
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd DLA-ICLR
   ```

### Setting up CRAS
2. **Configure API keys** in `cras/configs/api_config.json`
   ```bash
   # Edit the API configuration file
   nano cras/configs/api_config.json
   ```

3. **Run evaluations**: Use CRAS to evaluate agent outputs
   ```bash
   python cras/evaluate_scores.py 
   ```

### Setting up SAIL
4. **Create conda environment** and install dependencies
   ```bash
   cd sail
   conda env create -f sail_environment.yml
   conda activate sail
   ```

5. **Train models**: Use SAIL to train/fine-tune language models
   ```bash
   cd sail
   bash train.sh
   ```



