# EU Values Survey LLM Integration

[![codecov](https://codecov.io/gh/Telefonica-Scientific-Research/EUValues/branch/main/graph/badge.svg?token=EUValues_token_here)](https://codecov.io/gh/Telefonica-Scientific-Research/EUValues)
[![CI](https://github.com/Telefonica-Scientific-Research/EUValues/actions/workflows/main.yml/badge.svg)](https://github.com/Telefonica-Scientific-Research/EUValues/actions/workflows/main.yml)

This project implements a comprehensive system for administering the European Values Study (ZA7500) survey using Large Language Models (LLMs) across multiple languages and instruction-tuned models.

## Overview

The EU Values Survey LLM Integration provides tools for:

- **Multilingual Survey Administration**: Support for 7 languages (English, Spanish, Italian, Czech, Hungarian, Serbian, Russian)
- **Multi-Model LLM Support**: Compatible with multiple instruction-tuned models:
  - Gemma 2 27B
  - Apertus (7B, 70B)
  - Qwen 3 30B A3B
  - EuroLLM
  - Salamandra
  - Minimistral-3
- **Jinja2-based Prompt Templates**: Model-specific and language-specific prompt rendering
- **Response Parsing & Validation**: Structured constraint enforcement for survey responses
- **Batch Processing**: Automated workflow for processing survey questions across language-model combinations

## Project Structure

```
.
├── query_survey_llm.py              # Main orchestration script with argument parsing
├── local_llm_questions.ipynb         # Jupyter notebook with complete workflow
├── prompts/                          # Jinja2 templates for survey prompts
│   └── survey_prompt/
│       ├── survey_prompts_final_answer.jinja2
│       ├── survey_prompts_final_answer_gemma27b.jinja2
│       ├── survey_prompts_final_answer_apertus.jinja2
│       ├── survey_prompts_final_answer_qwen3_30b.jinja2
│       ├── survey_prompts_final_answer_eurollm.jinja2
│       ├── survey_prompts_final_answer_salamandra.jinja2
│       └── survey_prompts_final_answer_minimistral3.jinja2
├── Surveys/                         # Original survey data (7 languages)
│   ├── ZA7500_q_gb.csv
│   ├── ZA7500_q_es.csv
│   ├── ZA7500_q_it.csv
│   ├── ZA7500_q_cz.csv
│   ├── ZA7500_q_hu.csv
│   ├── ZA7500_q_rs.csv
│   └── ZA7500_q_ru.csv
├── Surveys_parsed/                  # Parsed survey data (cleaned/standardized)
├── Surveys_responses/               # LLM-generated responses (CSV + JSON formats)
└── euvalues/                        # Python package
    ├── __init__.py
    ├── base.py
    ├── cli.py
    ├── __main__.py
    └── VERSION
```

## Installation

### From source

```bash
git clone https://github.com/Telefonica-Scientific-Research/EUValues.git
cd EUValues
pip install -e .
```

### From PyPI

```bash
pip install euvalues
```

## Requirements

```
numpy>=1.24.0
pandas>=2.0.0
requests>=2.28.0
jinja2>=3.1.0
```

## Usage

### Command Line

#### Basic Usage

Process survey questions for a specific language and model:

```bash
python query_survey_llm.py --language es --models apertus
```

#### With Custom Server Configuration

```bash
python query_survey_llm.py \
  --host 192.168.1.100 \
  --port 8000 \
  --languages es it en_gb \
  --models apertus gemma27b qwen3_30b \
  --csv-dir ./Surveys_parsed \
  --output-dir ./Surveys_responses \
  --timeout 60
```

### Command Line Arguments

- `--host`: LLM server hostname (default: `127.0.0.1`)
- `--port`: LLM server port (default: `10000`)
- `--languages`: Space-separated list of language codes (default: all 7)
- `--models`: Space-separated list of LLM model names (default: all 6)
- `--csv-dir`: Directory containing survey CSV files (default: `./Surveys_parsed`)
- `--output-dir`: Output directory for responses (default: `./Surveys_responses`)
- `--timeout`: Request timeout in seconds (default: `120`)

### Python API

```python
from query_survey_llm import load_template, query_llm

# Load a model-specific template
template = load_template("apertus")

# Render template with survey variables
prompt = template.render(
    language="es",
    question_id="Q1",
    question_text="¿Cuál es tu opinión sobre...?",
    variable="tolerance",
    option_text="Completamente de acuerdo",
    response_scale="1-5"
)

# Query LLM
response = query_llm(prompt, host="127.0.0.1", port=10000)
print(response)
```

### Jupyter Notebook

Open `local_llm_questions.ipynb` for an interactive workflow that includes:
- Survey data loading and exploration
- Template rendering with variable substitution
- LLM querying with error handling
- Response parsing and validation
- Results aggregation and analysis

## Prompt Template Architecture

Templates use Jinja2 with the following variables:

- `language`: Survey language (gb, es, it, cz, hu, rs, ru)
- `question_id`: Question identifier
- `question_text`: Survey question in target language
- `variable`: Research variable being measured
- `option_text`: Response option text
- `response_scale`: Format for allowed responses (e.g., "1-5" or "yes/no")

### Example Template Structure

```jinja2
{% set instruction_prompt %}
{% if language == "es" %}
Responde la siguiente pregunta de una encuesta sobre valores europeos:
{% elif language == "gb" %}
Answer the following question from a survey about European values:
{% endif %}

**Pregunta**: {{ question_text }}
**Variable**: {{ variable }}
**Escala de respuesta**: {{ response_scale }}

Final answer: @@<response>@@
{% endset %}

<|im_start|>user
{{ instruction_prompt }}<|im_end|>
<|im_start|>assistant
```

## LLM Server Integration

This project requires a compatible LLM server implementing the OpenAI `/v1/chat/completions` endpoint.

### Recommended Setup: llama.cpp Server

```bash
# Build llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Download a quantized model
wget https://huggingface.co/models/model.gguf

# Start server
./server -m model.gguf -ngl 33 --port 10000
```

### Docker Alternative

See `Containerfile` for containerized deployment options.

## Response Format

Responses are saved in both CSV and JSON formats:

### CSV Format
```csv
question_id,question_text,variable,language,model,response,timestamp
Q1,"¿Cuál es tu opinión...?",tolerance,es,apertus,"Completamente de acuerdo",2024-01-20 10:30:00
```

### JSON Format
```json
{
  "metadata": {
    "language": "es",
    "model": "apertus",
    "timestamp": "2024-01-20T10:30:00"
  },
  "responses": [
    {
      "question_id": "Q1",
      "question_text": "¿Cuál es tu opinión...?",
      "variable": "tolerance",
      "response": "Completamente de acuerdo"
    }
  ]
}
```

## Survey Data

The project includes data from the European Values Study (ZA7500):
- **Coverage**: 7 European countries with different languages
- **Questions**: Survey on European values, attitudes, and beliefs
- **Format**: CSV files with question text in native languages

### Available Languages

| Code | Language | File |
|------|----------|------|
| gb | English (British) | ZA7500_q_gb.csv |
| es | Spanish | ZA7500_q_es.csv |
| it | Italian | ZA7500_q_it.csv |
| cz | Czech | ZA7500_q_cz.csv |
| hu | Hungarian | ZA7500_q_hu.csv |
| rs | Serbian | ZA7500_q_rs.csv |
| ru | Russian | ZA7500_q_ru.csv |

## Performance & Constraint Enforcement

The system implements multiple strategies to ensure LLM compliance with survey response constraints:

1. **Numbered Selection with Examples** (⭐⭐⭐⭐⭐ highest reliability)
   - Forces selection from numbered options
   - Includes correct/incorrect format examples

2. **JSON Schema Constraints** (⭐⭐⭐⭐⭐)
   - Leverages LLM JSON mode when available
   - Strict schema validation

3. **Explicit List + Repetition** (⭐⭐⭐⭐)
   - Lists valid options multiple times
   - Works well for open-source models

4. **Few-Shot Examples** (⭐⭐⭐)
   - Demonstrates correct response format
   - Adds token overhead

## Citation

If you use this codebase in your research, please cite:

```bibtex
@software{euvalues2024,
  title={EU Values Survey LLM Integration},
  author={ELOQUENCE Project},
  organization={Telefonica-Scientific-Research},
  year={2024},
  url={https://github.com/Telefonica-Scientific-Research/EUValues}
}
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Support

For issues, questions, or suggestions, please open an issue on [GitHub Issues](https://github.com/Telefonica-Scientific-Research/EUValues/issues).

## Project Context

This work is part of the [ELOQUENCE Project](https://eloquence-ai.eu/) - European Language Understanding and Question Answering in a Converged European Research Space.
