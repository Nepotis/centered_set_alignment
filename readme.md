# Centered Set Inference for AI Alignment

This repository contains a proof-of-concept implementation of the Centered Set Inference framework for AI alignment, as described in the research paper "Centered Set Inference for AI Alignment".

## Overview

The Centered Set Inference Engine (CSIE) is a novel approach to AI alignment that focuses on directional alignment toward a "center" of core values, rather than enforcing fixed boundaries between acceptable and unacceptable outputs. This approach enables:

- **Continuous adaptation** to evolving values
- **Granular feedback** on multiple value dimensions
- **Trajectory-based evaluation** that considers movement toward or away from ideals
- **Improved long-term safety** through focus on core principles rather than specific rules

## Installation

```bash
# Clone the repository
git clone https://github.com/nepotis/centered-set-inference.git
cd centered-set-inference

# Install the package
pip install -e .
```

## Usage

### Running the Therapeutic Chatbot Demo

The simplest way to try out the framework is to run the therapeutic chatbot demo:

```bash
python -m centered_set_inference.main --chatbot
```

This will:
1. Generate synthetic training data if it doesn't exist
2. Load or train the alignment head
3. Launch a Gradio interface for interacting with the chatbot

### Training the Alignment Head

To explicitly train or retrain the alignment head:

```bash
python -m centered_set_inference.main --retrain --epochs 5
```

### Running Benchmarks

To evaluate the framework on test prompts:

```bash
python -m centered_set_inference.main --benchmark
```

This will generate visualizations and statistics in the `results/` directory.

### Advanced Options

```bash
python -m centered_set_inference.main --help
```

## Framework Components

- **ValueCenter**: Defines the core values and their relative weights
- **AlignmentHead**: Neural network that evaluates outputs along value dimensions
- **CenteredSetInferenceEngine**: Main engine that guides generation toward the center

## Example Use Cases

The current implementation focuses on a therapeutic chatbot, but the framework can be extended to other domains such as:

- Educational tutoring
- Collaborative brainstorming
- Coding assistance

## Limitations

This is a proof-of-concept implementation with several limitations:

- Uses synthetic data rather than human-annotated examples
- Limited to smaller models due to computational constraints
- Simplified value dimensions compared to a production system

## Citation

If you use this code in your research, please cite:

```
@article{wiedeman2025centered,
  title={Centered Set Inference for AI Alignment},
  author={Wiedeman, Gregory M.},
  journal={Forthcoming},
  year={2025}
}
```

## License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.
