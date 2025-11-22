import numpy as np
import logging
import os
import time
import json
from llm_client import PromptManager
from tqdm.auto import tqdm
from utils import (
    clean_response, alphabetize_csv_text, get_csv_text_n_entries, load_required_files,
    save_memory, load_files_with_optional, save_memory_without_metadata, save_individual_metadata
)

logger = logging.getLogger(__name__)


# ===================== QA SUPPORT ===================== #

def save_with_qa(args, llm_client, content, step_name, filename, metadata,
                 context=None, context_type=None):
    """Run QA and save content with QA results embedded in metadata."""
    original_content = content
    qa_passed, qa_data, final_content = run_qa_step(
        args, llm_client, step_name, content, step_name, context=context, context_type=context_type
    )

    if final_content != content:
        logger.info(f"Using amended {step_name} content from QA")
        content = final_content

    if not qa_passed:
        logger.warning(f"{step_name} QA failed (continuing). Issues: {qa_data.get('issues') if qa_data else 'N/A'}")

    qa_metadata = qa_data if qa_data else {}
    qa_metadata['content_before_qa'] = original_content
    qa_metadata['content_after_qa'] = final_content
    qa_metadata['content_changed'] = original_content != final_content
    metadata['qa_results'] = qa_metadata

    step_memory_dir = os.path.join(args.memory_dir, step_name)
    save_memory(content, step_memory_dir, filename, metadata)

    if qa_data:
        qa_filename = f"{step_name}_qa.json"
        qa_filepath = os.path.join(step_memory_dir, qa_filename)
        qa_data_with_content = qa_metadata.copy()

        if args.continue_qa and os.path.exists(qa_filepath):
            try:
                with open(qa_filepath, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
                if 'all_iterations' in existing and 'all_iterations' in qa_data:
                    max_iter = max((it.get('iteration', 0) for it in existing['all_iterations']), default=0)
                    for it in qa_data['all_iterations']:
                        it['iteration'] += max_iter
                    existing['all_iterations'].extend(qa_data['all_iterations'])
                    existing['final_qa'] = qa_data.get('final_qa')
                    existing['continue_qa_run'] = True
                    existing.update({
                        'content_before_qa': original_content,
                        'content_after_qa': final_content,
                        'content_changed': original_content != final_content,
                    })
                    qa_data_with_content = existing
            except Exception as e:
                logger.warning(f"Failed extending QA file, recreating: {e}")

        save_memory(json.dumps(qa_data_with_content, indent=2, ensure_ascii=False), step_memory_dir, qa_filename, {})

    return content


def run_qa_step(args, llm_client, step_name, content, content_type="phonology", context=None, context_type=None):
    """Run QA critic/amend loop for a content artifact."""
    if context and not context_type:
        raise ValueError("Missing context_type")
    if context_type and not context:
        raise ValueError("Missing context")
    has_context = context is not None

    if not getattr(args, 'qa_enabled', False):
        return True, None, content

    prompt_dir = os.path.join(args.prompt_dir, 'qa')
    if step_name == 'translation':
        if has_context:
            critic = PromptManager.load_prompt(os.path.join(prompt_dir, 'qa_translation_critic_with_context.txt'))
        else:
            critic = PromptManager.load_prompt(os.path.join(prompt_dir, 'qa_critic.txt'))
        amend = PromptManager.load_prompt(os.path.join(prompt_dir, 'qa_translation_amend.txt'))
    else:
        if has_context:
            critic = PromptManager.load_prompt(os.path.join(prompt_dir, 'qa_critic_with_context.txt'))
        else:
            critic = PromptManager.load_prompt(os.path.join(prompt_dir, 'qa_critic.txt'))
        amend = PromptManager.load_prompt(os.path.join(prompt_dir, 'qa_amend.txt'))

    current = content
    all_iters = []
    final_qa = None
    max_iters = getattr(args, 'self_refine_steps', 3)

    for i in range(max_iters):
        if has_context:
            qa_prompt = PromptManager.format_prompt(critic, content=current, content_type=content_type, context=context, context_type=context_type)
        else:
            qa_prompt = PromptManager.format_prompt(critic, content=current, content_type=content_type)
        logger.info(f"QA prompt ({step_name} iter {i+1}): {qa_prompt}")
        _, qa_raw = llm_client.generate_and_extract(qa_prompt, do_sleep=False)
        qa_raw = clean_response(qa_raw, 'json')
        try:
            qa_data = json.loads(qa_raw)
            final_qa = qa_data
            overall = qa_data.get('overall_score', 0)
            iter_record = {'iteration': i+1, 'qa_data': qa_data, 'content_length': len(current), 'amended': False}
            # Determine threshold
            if getattr(args, 'qa_threshold', None) is not None:
                threshold = args.qa_threshold
            else:
                if step_name == 'phonology':
                    threshold = args.qa_threshold_phonology
                elif step_name == 'grammar':
                    threshold = args.qa_threshold_grammar
                elif step_name == 'translation':
                    threshold = args.qa_threshold_translation
                elif step_name == 'lexicon':
                    threshold = args.qa_threshold_lexicon
                else:
                    threshold = 8.0
            if overall >= threshold:
                all_iters.append(iter_record)
                return True, {'final_qa': qa_data, 'all_iterations': all_iters}, current
            if i < max_iters - 1:
                iter_record['amended'] = True
                all_iters.append(iter_record)
                amend_prompt = PromptManager.format_prompt(amend, content=current, judgement=qa_raw)
                _, amended = llm_client.generate_and_extract(amend_prompt, do_sleep=False)
                current = amended
            else:
                all_iters.append(iter_record)
        except json.JSONDecodeError:
            iter_record = {'iteration': i+1, 'qa_data': None, 'error': 'json_parse_failed', 'raw_response': qa_raw, 'amended': False}
            all_iters.append(iter_record)
            if i == max_iters - 1:
                return False, {'final_qa': None, 'all_iterations': all_iters}, current

    return False, {'final_qa': final_qa, 'all_iterations': all_iters}, current


# ===================== GENERATION HELPERS ===================== #

def _generate_with_prompts(llm_client, prompts, kwargs_list, do_sleep_flags=None):
    if do_sleep_flags is None:
        do_sleep_flags = [True] * len(prompts)
    responses = []
    for i, (prompt_key, kwargs) in enumerate(zip(prompts.keys(), kwargs_list)):
        prompt = PromptManager.format_prompt(prompts[prompt_key], **kwargs)
        logger.info(f"Prompt {i+1}: {prompt}")
        full_response, extracted = llm_client.generate_and_extract(
            prompt, do_sleep=do_sleep_flags[i] if i < len(do_sleep_flags) else False
        )
        responses.append((full_response, extracted))
    return responses


# ===================== PHONOLOGY ===================== #

def run_phonology_step(args, llm_client):
    if args.continue_qa:
        existing_file = os.path.join(args.memory_dir, 'phonology', 'phonology.txt')
        if os.path.exists(existing_file):
            with open(existing_file, 'r', encoding='utf-8') as f:
                content = f.read()
            metadata = {'continue_qa': True}
            save_with_qa(args, llm_client, content, 'phonology', 'phonology.txt', metadata)
            return True
    prompt_dir = os.path.join(args.prompt_dir, 'phonology')
    prompts = PromptManager.load_prompts(prompt_dir, ['phon_step1_checklist.txt', 'phon_step2_summary.txt', 'phon_step3_word_shapes.txt'])
    custom = "(none)" if args.custom_constraints is None else args.custom_constraints
    values = np.random.randint(args.phon_n_answers, size=args.phon_n_questions) + 1
    values = [int(x) for x in values]
    kwargs_list = [{'n_questions': args.phon_n_questions, 'n_answers': args.phon_n_answers, 'scale_size': args.phon_scale_size}]
    responses = _generate_with_prompts(llm_client, {'step1': prompts['phon_step1_checklist']}, kwargs_list)
    _, checklist = responses[0]
    step2_kwargs = {'checklist': checklist, 'values': str(list(values)), 'custom': custom}
    step2_responses = _generate_with_prompts(llm_client, {'step2': prompts['phon_step2_summary']}, [step2_kwargs], [False])
    _, phonology = step2_responses[0]
    step3_kwargs = {'phonology': phonology, 'n': args.phon_n_words, 'custom': custom}
    step3_responses = _generate_with_prompts(llm_client, {'step3': prompts['phon_step3_word_shapes']}, [step3_kwargs], [False])
    _, word_shapes = step3_responses[0]
    full = phonology.strip() + '\n\n' + word_shapes.strip()
    metadata = {**kwargs_list[0], **step2_kwargs, **step3_kwargs}
    save_with_qa(args, llm_client, full, 'phonology', 'phonology.txt', metadata)
    return True


# ===================== GRAMMAR ===================== #

def run_grammar_step(args, llm_client):
    if args.continue_qa:
        existing_file = os.path.join(args.memory_dir, 'grammar', 'grammar.txt')
        if os.path.exists(existing_file):
            with open(existing_file, 'r', encoding='utf-8') as f:
                merged = f.read()
            files = load_required_files(args.memory_dir, {'phonology': 'phonology.txt'})
            if files is None:
                return False
            save_with_qa(args, llm_client, merged, 'grammar', 'grammar.txt', {'continue_qa': True}, context=files['phonology'], context_type='phonology')
            return True
    files = load_required_files(args.memory_dir, {'phonology': 'phonology.txt'})
    if files is None:
        return False
    phonology = files['phonology']
    prompt_dir = os.path.join(args.prompt_dir, 'grammar')
    prompts = PromptManager.load_prompts(prompt_dir, ['gram_step1_checklist.txt', 'gram_step2_summary.txt', 'gram_step3_expand.txt', 'merge_sections.txt'])
    custom = "(none)" if args.custom_constraints is None else args.custom_constraints
    values = np.random.randint(args.gram_n_answers, size=args.gram_n_questions) + 1
    values = [int(x) for x in values]
    kwargs_list = [{'n_questions': args.gram_n_questions, 'n_answers': args.gram_n_answers, 'scale_size': args.gram_scale_size}]
    step1 = _generate_with_prompts(llm_client, {'step1': prompts['gram_step1_checklist']}, kwargs_list)
    _, checklist = step1[0]
    step2_kwargs = {'checklist': checklist, 'values': str(list(values)), 'custom': custom, 'phonology': phonology}
    step2 = _generate_with_prompts(llm_client, {'step2': prompts['gram_step2_summary']}, [step2_kwargs])
    _, grammar = step2[0]
    step3_kwargs = {'grammar': grammar, 'custom': custom, 'phonology': phonology}
    step3 = _generate_with_prompts(llm_client, {'step3': prompts['gram_step3_expand']}, [step3_kwargs])
    _, expanded = step3[0]
    summaries = f"===SUMMARY 1:===\n{grammar}\n\n===SUMMARY 2:===\n{expanded}\n===END SUMMARIES==="
    step4_kwargs = {'summaries': summaries}
    step4 = _generate_with_prompts(llm_client, {'step4': prompts['merge_sections']}, [step4_kwargs], [False])
    _, merged = step4[0]
    metadata = {**kwargs_list[0], **step2_kwargs, **step3_kwargs}
    save_with_qa(args, llm_client, merged, 'grammar', 'grammar.txt', metadata, context=phonology, context_type='phonology')
    return True


# ===================== LEXICON (CSV + QA) ===================== #

def _run_iterative_csv_step_with_qa(args, llm_client, step_name, required_files, prompt_files,
                                   min_entries_attr, n_per_iter_attr, extra_sleep_attr, max_iters_attr):
    files = load_required_files(args.memory_dir, required_files)
    if files is None:
        return False
    prompt_dir = os.path.join(args.prompt_dir, step_name)
    prompts = PromptManager.load_prompts(prompt_dir, prompt_files)
    p1 = prompt_files[0].replace('.txt', '')
    p2 = prompt_files[1].replace('.txt', '')
    min_entries = getattr(args, min_entries_attr)
    n_per_iter = getattr(args, n_per_iter_attr)
    extra_sleep = getattr(args, extra_sleep_attr)
    max_iters = getattr(args, max_iters_attr)
    pbar = tqdm(desc=f"Making {step_name}", total=min_entries)
    step1_kwargs = {k: v for k, v in files.items() if k in ['phonology', 'grammar']}
    step1 = _generate_with_prompts(llm_client, {'step1': prompts[p1]}, [step1_kwargs])
    _, csv_raw = step1[0]
    csv_data = clean_response(csv_raw, 'csv')
    csv_data = alphabetize_csv_text(csv_data)
    pbar.update(get_csv_text_n_entries(csv_data))
    i = 0
    while get_csv_text_n_entries(csv_data) < min_entries and i < max_iters:
        logger.info(f"Extra sleep for {step_name}: {extra_sleep}s")
        if not args.debug:
            time.sleep(extra_sleep)
        i += 1
        pbar.set_description(f"Making {step_name} (expansion iter {i}/{max_iters})")
        step2_kwargs = {**files, step_name: csv_data, 'n': n_per_iter}
        step2 = _generate_with_prompts(llm_client, {'step2': prompts[p2]}, [step2_kwargs])
        _, expanded_raw = step2[0]
        expanded = clean_response(expanded_raw, 'csv')
        pbar.update(get_csv_text_n_entries(expanded))
        csv_data = csv_data.strip() + '\n' + '\n'.join(expanded.strip().splitlines()[1:])
        csv_data = alphabetize_csv_text(csv_data)
    if i >= max_iters and get_csv_text_n_entries(csv_data) < min_entries:
        logger.warning(f"{step_name} hit max iterations with {get_csv_text_n_entries(csv_data)} entries (target {min_entries})")
    # Convert to text for QA
    text_version = _csv_to_text_for_qa(csv_data)
    context = f"Phonology:\n{files['phonology']}\n\nGrammar:\n{files['grammar']}"
    metadata = {'min_entries': min_entries, 'n_per_iter': n_per_iter, 'max_iters': max_iters, 'actual_entries': get_csv_text_n_entries(csv_data), 'iterations_used': i}
    final_text = save_with_qa(args, llm_client, text_version, step_name, f'{step_name}.csv', metadata, context=context, context_type='phonology and grammar')
    final_csv = _text_to_csv_for_qa(final_text)
    step_memory_dir = os.path.join(args.memory_dir, step_name)
    with open(os.path.join(step_memory_dir, f'{step_name}.csv'), 'w', encoding='utf-8') as f:
        f.write(final_csv)
    qa_file = os.path.join(step_memory_dir, f'{step_name}_qa.json')
    if os.path.exists(qa_file):
        try:
            with open(qa_file, 'r', encoding='utf-8') as f:
                qa_data = json.load(f)
            qa_data['content_before_qa_csv'] = csv_data
            qa_data['content_after_qa_csv'] = final_csv
            qa_data['csv_content_changed'] = csv_data != final_csv
            with open(qa_file, 'w', encoding='utf-8') as f:
                json.dump(qa_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Could not augment QA file with CSV versions: {e}")
    return True


def _csv_to_text_for_qa(csv_data: str) -> str:
    lines = csv_data.strip().split('\n')
    if not lines:
        return ''
    header = lines[0]
    out = [f"Lexicon entries (format: {header}):"]
    for idx, line in enumerate(lines[1:], 1):
        if line.strip():
            out.append(f"{idx}. {line}")
    return '\n'.join(out)


def _text_to_csv_for_qa(text_content: str) -> str:
    lines = text_content.strip().split('\n')
    csv_lines = []
    header_added = False
    for line in lines:
        l = line.strip()
        if not l:
            continue
        if '.' in l and ',' in l:
            parts = l.split('.', 1)
            if len(parts) > 1:
                csv_entry = parts[1].strip()
                if not header_added:
                    cols = len(csv_entry.split(','))
                    if cols == 2:
                        csv_lines.append('word,translation')
                    elif cols == 3:
                        csv_lines.append('word,translation,pos')
                    else:
                        csv_lines.append('word,translation,pos,notes')
                    header_added = True
                csv_lines.append(csv_entry)
        elif ',' in l and not l.startswith('Lexicon') and not l.startswith('format:'):
            if not header_added:
                csv_lines.append(l)
                header_added = True
            else:
                csv_lines.append(l)
    if csv_lines and not header_added:
        csv_lines.insert(0, 'word,translation')
    return '\n'.join(csv_lines)


def run_lexicon_step(args, llm_client):
    if args.continue_qa:
        existing = os.path.join(args.memory_dir, 'lexicon', 'lexicon.csv')
        if os.path.exists(existing):
            with open(existing, 'r', encoding='utf-8') as f:
                csv_data = f.read()
            text_version = _csv_to_text_for_qa(csv_data)
            files = load_required_files(args.memory_dir, {'phonology': 'phonology.txt', 'grammar': 'grammar.txt'})
            if files is None:
                return False
            context = f"Phonology:\n{files['phonology']}\n\nGrammar:\n{files['grammar']}"
            final_text = save_with_qa(args, llm_client, text_version, 'lexicon', 'lexicon.csv', {'continue_qa': True}, context=context, context_type='phonology and grammar')
            final_csv = _text_to_csv_for_qa(final_text)
            step_memory_dir = os.path.join(args.memory_dir, 'lexicon')
            with open(os.path.join(step_memory_dir, 'lexicon.csv'), 'w', encoding='utf-8') as f:
                f.write(final_csv)
            qa_file = os.path.join(step_memory_dir, 'lexicon_qa.json')
            if os.path.exists(qa_file):
                try:
                    with open(qa_file, 'r', encoding='utf-8') as f:
                        qa_data = json.load(f)
                    qa_data['content_before_qa_csv'] = csv_data
                    qa_data['content_after_qa_csv'] = final_csv
                    qa_data['csv_content_changed'] = csv_data != final_csv
                    with open(qa_file, 'w', encoding='utf-8') as f:
                        json.dump(qa_data, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    logger.warning(f"Could not update lexicon QA file: {e}")
            return True
    required_files = {'phonology': 'phonology.txt', 'grammar': 'grammar.txt'}
    prompt_files = ['lex_step1_extract.txt', 'lex_step2_expand.txt']
    return _run_iterative_csv_step_with_qa(args, llm_client, 'lexicon', required_files, prompt_files, 'lexicon_min_entries', 'lexicon_n_per_iter', 'lexicon_extra_sleep', 'lexicon_max_iters')


# ===================== TRANSLATION (single) ===================== #

def run_translation_step(args, llm_client):
    if args.continue_qa:
        existing = os.path.join(args.memory_dir, 'translation', 'translation.json')
        if os.path.exists(existing):
            with open(existing, 'r', encoding='utf-8') as f:
                content = f.read()
            files = load_required_files(args.memory_dir, {'phonology': 'phonology.txt', 'grammar': 'grammar.txt'})
            if files is None:
                return False
            context = f"PHONOLOGY:\n{files['phonology']}\n\nGRAMMAR:\n{files['grammar']}"
            save_with_qa(args, llm_client, content, 'translation', 'translation.json', {'continue_qa': True, 'input_sentence': args.translation_input_sentence}, context=context, context_type='language_spec')
            return True
    required = {'phonology': 'phonology.txt', 'grammar': 'grammar.txt'}
    optional = {'lexicon': 'lexicon.csv'}
    files = load_files_with_optional(args.memory_dir, required, optional)
    if files is None:
        return False
    prompt_dir = os.path.join(args.prompt_dir, 'translation')
    raw_prompt = PromptManager.load_prompt(os.path.join(prompt_dir, 'translation_single.txt'))
    if 'lexicon' in files:
        lex_section = f"""It has the following lexicon:\n\n=== START ===\n{files['lexicon']}\n=== END ==="""
    else:
        lex_section = """Note: No specific lexicon has been provided. You will need to create appropriate vocabulary words that follow the phonological and morphological patterns of the language."""
    kwargs = {'phonology': files['phonology'], 'grammar': files['grammar'], 'lexicon_section': lex_section, 'input_sentence': args.translation_input_sentence}
    prompt = PromptManager.format_prompt(raw_prompt, **kwargs)
    _, content = llm_client.generate_and_extract(prompt, do_sleep=False)
    content = clean_response(content, 'json')
    context = f"PHONOLOGY:\n{files['phonology']}\n\nGRAMMAR:\n{files['grammar']}"
    if 'lexicon' in files:
        context += f"\n\nLEXICON:\n{files['lexicon']}"
    metadata = {'input_sentence': args.translation_input_sentence, 'lexicon_available': 'lexicon' in files}
    save_with_qa(args, llm_client, content, 'translation', 'translation.json', metadata, context=context, context_type='language_spec')
    return True