#!/usr/bin/env python3
"""
ConlangCrafter: AI-Powered Constructed Language Generator

This script generates constructed languages (conlangs) using AI models.
It supports phonology, grammar, lexicon generation, and translation.
"""

import os
import sys
import time
import logging
import uuid
import json
from argparse import ArgumentParser
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from llm_client import LLMClientGemini, LLMClientDeepseek, LLMClientOpenAI
from pipeline_steps import run_phonology_step, run_grammar_step, run_lexicon_step, run_translation_step

logger = logging.getLogger(__name__)


def generate_language_id():
    """Generate a unique language ID."""
    return str(uuid.uuid4())[:8]


def setup_directories(output_dir, language_id):
    """Set up directories for a specific language."""
    lang_dir = os.path.join(output_dir, 'languages', language_id)
    memory_dir = os.path.join(lang_dir, 'memory')
    logs_dir = os.path.join(lang_dir, 'logs')
    
    os.makedirs(lang_dir, exist_ok=True)
    os.makedirs(memory_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    return lang_dir, memory_dir, logs_dir


def setup_logging(output_file: str):
    """Set up logging configuration."""
    logdir = os.path.dirname(output_file)
    os.makedirs(logdir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_file),
            logging.StreamHandler()
        ]
    )


def save_metadata(lang_dir, language_id, args):
    """Save metadata about the language generation."""
    metadata = {
        'language_id': language_id,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model': args.model,
        'steps': args.steps.split(','),
        'custom_constraints': args.custom_constraints,
        'parameters': {
            'temperature': getattr(args, 'temperature', None),
            'max_tokens': getattr(args, 'max_tokens', None),
        }
    }
    
    metadata_file = os.path.join(lang_dir, 'metadata.json')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata_file


def run_analysis_step(args, llm_client):
    """Run final analysis on the complete language."""
    logger.info("Starting final language analysis")
    
    files = load_required_files(args.memory_dir, {
        'phonology': 'phonology.txt',
        'grammar': 'grammar.txt',
        'lexicon': 'lexicon.csv'
    })
    if files is None:
        logger.error("Could not load required files for analysis")
        return
        
    prompt_dir = os.path.join(args.prompt_dir, 'analysis')
    try:
        prompt = PromptManager.load_prompt(os.path.join(prompt_dir, 'feature_analysis.txt'))
    except Exception as e:
        logger.error(f"Could not load analysis prompt: {e}")
        return False
    
    kwargs = {
        'phonology': files['phonology'],
        'grammar': files['grammar'],
        'lexicon': files['lexicon']
    }
    
    logger.info("Generating final language analysis")
    _, analysis = llm_client.generate_and_extract(
        PromptManager.format_prompt(prompt, **kwargs),
        do_sleep=False
    )
    
    if analysis is None:
        logger.error("Failed to generate analysis")
        return
        
    # Save analysis
    analysis_dir = os.path.join(args.memory_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    with open(os.path.join(analysis_dir, 'analysis.txt'), 'w', encoding='utf-8') as f:
        f.write(analysis)
    
    logger.info("Final analysis completed and saved")


def get_args():
    """Parse command line arguments."""
    parser = ArgumentParser(description='Generate constructed languages using AI')
    
    # Model settings
    parser.add_argument(
        '--model',
        default='gemini-2.5-pro',
        help=(
            'Model identifier to use. Examples: gemini-2.5-pro, gemini-1.5-flash, '
            'o4-mini, gpt-4o, gpt-5, deepseek-ai/DeepSeek-R1. '
            'Any valid provider model string is accepted.'
        ),
    )
    parser.add_argument('--max-tokens', type=int, default=8192,
                       help='Maximum tokens for generation')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Temperature for sampling')
    parser.add_argument('--thinking-budget', type=int, default=1000,
                       help='Thinking budget for models that support it')
    parser.add_argument('--reasoning-effort', default='medium', choices=['low','medium','high'],
                       help='Reasoning effort for OpenAI o-series')
    parser.add_argument('--sleep-between-calls', type=float, default=30,
                       help='Sleep time between API calls (seconds)')

    # QA settings
    parser.add_argument('--qa-enabled', action='store_true',
                        help='Enable QA self-refine (critic/amend) loop for supported steps')
    parser.add_argument('--self-refine-steps', type=int, default=3,
                        help='Number of QA self-refine (critic/amend) cycles')
    parser.add_argument('--run-analysis', action='store_true',
                        help='Run final analysis on the complete language')
    parser.add_argument('--qa-threshold', type=float, default=None,
                        help='Global passing score threshold (1–10 scale) overriding all per-step thresholds if set')
    parser.add_argument('--qa-threshold-phonology', type=float, default=8.0,
                        help='Passing score threshold (1–10 scale) for phonology QA')
    parser.add_argument('--qa-threshold-grammar', type=float, default=8.0,
                        help='Passing score threshold (1–10 scale) for grammar QA')
    parser.add_argument('--qa-threshold-lexicon', type=float, default=8.0,
                        help='Passing score threshold (1–10 scale) for lexicon QA')
    parser.add_argument('--qa-threshold-translation', type=float, default=8.0,
                        help='Passing score threshold (1–10 scale) for translation QA')
    parser.add_argument('--continue-qa', action='store_true',
                        help='If a QA report exists, continue from previous iterations and append results')
    
    # Pipeline settings
    parser.add_argument('--steps', default='phonology,grammar,lexicon,translation',
                       help='Comma-separated list of steps to run')
    parser.add_argument('--custom-constraints', 
                       help='Custom constraints for language generation')
    parser.add_argument('--translation-sentence', 
                       default='The quick brown fox jumps over the lazy dog.',
                       help='Sentence to translate into the constructed language')
    
    # Generation parameters
    parser.add_argument('--phon-n-questions', type=int, default=10,
                       help='Number of phonology questions')
    parser.add_argument('--phon-n-answers', type=int, default=5,
                       help='Number of phonology answer options')
    parser.add_argument('--phon-scale-size', type=int, default=5,
                       help='Phonology scale size')
    parser.add_argument('--phon-n-words', type=int, default=25,
                       help='Number of phonology word examples')
    
    parser.add_argument('--gram-n-questions', type=int, default=10,
                       help='Number of grammar questions')
    parser.add_argument('--gram-n-answers', type=int, default=5,
                       help='Number of grammar answer options')
    parser.add_argument('--gram-scale-size', type=int, default=5,
                       help='Grammar scale size')
    
    parser.add_argument('--lexicon-min-entries', type=int, default=50,
                       help='Minimum lexicon entries')
    parser.add_argument('--lexicon-n-per-iter', type=int, default=15,
                       help='Lexicon entries per iteration')
    parser.add_argument('--lexicon-max-iters', type=int, default=5,
                       help='Maximum lexicon iterations')
    parser.add_argument('--lexicon-extra-sleep', type=float, default=30,
                       help='Extra sleep for lexicon generation')
    
    # Paths
    parser.add_argument('--prompt-dir', default='prompts',
                       help='Directory containing prompt templates')
    parser.add_argument('--output-dir', default='output',
                       help='Output directory for generated languages')
    
    # Debug mode
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with dummy responses')
    
    return parser.parse_args()


def main():
    """Main function to run the ConlangCrafter pipeline."""
    args = get_args()
    
    # Generate language ID
    language_id = generate_language_id()
    print(f"Generating language with ID: {language_id}")
    
    # Set up directories
    lang_dir, memory_dir, logs_dir = setup_directories(args.output_dir, language_id)
    args.memory_dir = memory_dir

    # Write the generated language id to a file so external callers can find it
    try:
        last_id_file = os.path.join(args.output_dir, 'LAST_LANGUAGE_ID')
        with open(last_id_file, 'w', encoding='utf-8') as f:
            f.write(language_id)
        logger.debug(f"Wrote LAST_LANGUAGE_ID to {last_id_file}")
    except Exception as e:
        logger.warning(f"Could not write LAST_LANGUAGE_ID file: {e}")
    
    # Set up logging
    log_file = os.path.join(logs_dir, 'pipeline.log')
    setup_logging(log_file)
    
    logger.info(f"Starting language generation with ID: {language_id}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Steps: {args.steps}")
    
    # Save metadata
    metadata_file = save_metadata(lang_dir, language_id, args)
    logger.info(f"Metadata saved to: {metadata_file}")
    
    # Initialize LLM client
    if args.model.startswith('gemini'):
        llm_client = LLMClientGemini(
            model_checkpoint=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            sleep_between_calls=args.sleep_between_calls,
            debug=args.debug,
            thinking_budget=args.thinking_budget
        )
    elif args.model.startswith('deepseek'):
        llm_client = LLMClientDeepseek(
            model_checkpoint=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            sleep_between_calls=args.sleep_between_calls,
            debug=args.debug
        )
    elif args.model.startswith('o') or args.model.startswith('gpt-'):
        # OpenAI client supports both o-series reasoning models and gpt-4o style
        llm_client = LLMClientOpenAI(
            model_checkpoint=args.model,
            max_tokens=args.max_tokens,
            reasoning_effort=args.reasoning_effort,
            temperature=args.temperature,
            sleep_between_calls=args.sleep_between_calls,
            debug=args.debug
        )
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    # Parse and run steps
    steps = [step.strip() for step in args.steps.split(',')]
    step_functions = {
        'phonology': run_phonology_step,
        'grammar': run_grammar_step,
        'lexicon': run_lexicon_step,
        'translation': run_translation_step,
    }
    
    print(f"\nRunning steps: {', '.join(steps)}")
    
    for i, step in enumerate(steps):
        if step not in step_functions:
            logger.error(f"Unknown step: {step}")
            continue
        
        print(f"\n=== Running {step} step ({i+1}/{len(steps)}) ===")
        logger.info(f"Starting {step} step")
        
        # Add translation sentence to args
        if step == 'translation':
            args.translation_input_sentence = args.translation_sentence
        
        try:
            result = step_functions[step](args, llm_client)
            if not result:
                logger.error(f"Step {step} failed")
                break
            logger.info(f"Completed {step} step")
        except Exception as e:
            logger.error(f"Error in {step} step: {e}")
            break
        
        # Sleep between steps
        if i < len(steps) - 1 and not args.debug:
            logger.info(f"Sleeping for 30 seconds...")
            time.sleep(30)
    
    print(f"\nLanguage generation completed!")
    print(f"Results saved in: {lang_dir}")
    logger.info(f"Language generation completed for ID: {language_id}")
    
    # Run final analysis if enabled
    if args.run_analysis:
        logger.info("Running final language analysis...")
        try:
            run_analysis_step(args, llm_client)
            print("Final analysis completed successfully.")
            logger.info("Final analysis completed successfully.")
        except Exception as e:
            print(f"Error during analysis: {e}")
            logger.error(f"Error during analysis: {e}")


if __name__ == '__main__':
    main()