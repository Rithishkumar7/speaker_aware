#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""WhisperX ASR pipeline for MISP-Meeting dev-F8N data."""
import argparse
import gc
import os
import re

import torch
import tqdm
import whisperx
import zhconv

import tool

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, 'dev-F8N')
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output', 'dev-F8N')

NOISE_MARKERS = {
    '<NOISE>', '<knock>', '<OVERCLEAR>', '<DEAF>', '<moving  sound>',
    '<laughter>', '<sound of door>', '<其他说话人>', '<主说话人>', '<非会议内容>',
}


def find_wav_files(data_dir):
    return sorted(tool.find_all_target_files(data_dir, '*.wav'))


def parse_wav_meta(wav_path, data_dir):
    rel = os.path.relpath(wav_path, data_dir)
    parts = rel.replace('\\', '/').split('/')
    session = parts[0]
    basename = os.path.splitext(os.path.basename(wav_path))[0]
    clip_id = basename.split('-')[-1]
    return session, clip_id, basename


def transcribe_wav(model, wav_path, device, batch_size, language):
    audio = whisperx.load_audio(wav_path)
    result = model.transcribe(
        audio,
        batch_size=batch_size,
        language=language,
    )
    align_model, metadata = whisperx.load_align_model(
        language_code=language,
        device=device,
    )
    result = whisperx.align(
        result['segments'],
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )
    del align_model
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    return result


def segments_to_lines(segments, speaker_label='1'):
    lines = []
    for seg in segments:
        text = seg.get('text', '').strip()
        if not text:
            continue
        start = round(float(seg['start']), 2)
        end = round(float(seg['end']), 2)
        lines.append((start, end, speaker_label, zhconv.convert(text, 'zh-hans')))
    return lines


def build_meeting_text(line_items):
    sorted_lines = sorted(line_items, key=lambda x: (x[0], x[1]))
    meeting_content = ''
    for start, end, speaker, text in sorted_lines:
        meeting_content += '{}-{} SPK{}: {}\n'.format(start, end, speaker, text)
    return meeting_content.strip()


def run_recognition(data_dir, output_dir, model_name, language, batch_size, device, sessions=None):
    os.makedirs(output_dir, exist_ok=True)
    compute_type = 'float16' if device == 'cuda' else 'int8'
    download_root = os.path.join(output_dir, 'whisperx_models')

    wav_files = find_wav_files(data_dir)
    if sessions:
        wav_files = [p for p in wav_files if parse_wav_meta(p, data_dir)[0] in sessions]
    if not wav_files:
        raise FileNotFoundError('No wav files found under {}'.format(data_dir))

    model = whisperx.load_model(
        model_name,
        device,
        compute_type=compute_type,
        language=language,
        download_root=download_root,
    )

    recognition_dict = {}
    session_lines = {}

    for wav_path in tqdm.tqdm(wav_files, desc='WhisperX'):
        session, clip_id, key = parse_wav_meta(wav_path, data_dir)
        result = transcribe_wav(model, wav_path, device, batch_size, language)
        full_text = ' '.join(
            zhconv.convert(seg.get('text', '').strip(), 'zh-hans')
            for seg in result.get('segments', [])
            if seg.get('text', '').strip()
        )
        recognition_dict[key] = {
            'wav': wav_path,
            'session': session,
            'clip_id': clip_id,
            'result': full_text,
            'segments': result.get('segments', []),
        }
        session_lines.setdefault(session, []).extend(
            segments_to_lines(result.get('segments', []), speaker_label=clip_id)
        )

    del model
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    recognition_json = os.path.join(
        output_dir, 'recognition_by_whisperx_{}.json'.format(model_name.replace('/', '-')))
    tool.json2dic(recognition_json, recognition_dict)

    meeting_dict = {
        session: build_meeting_text(lines)
        for session, lines in session_lines.items()
    }
    meeting_json = os.path.join(
        output_dir, 'recognition_by_whisperx_{}_meeting.json'.format(model_name.replace('/', '-')))
    tool.json2dic(meeting_json, meeting_dict)

    for session, content in meeting_dict.items():
        txt_path = os.path.join(output_dir, 'meeting_{}.txt'.format(session))
        with open(txt_path, 'w', encoding='utf-8') as handle:
            handle.write(content)

    print('Saved recognition:', recognition_json)
    print('Saved meetings:', meeting_json)
    return recognition_json, meeting_json


def build_reference_from_textgrid(data_dir, output_dir):
    """Optional: build reference meeting transcripts from TextGrid labels."""
    textgrid_files = sorted(tool.find_all_target_files(data_dir, '*.TextGrid'))
    session_lines = {}
    skip_pattern = re.compile(r'^(\*+|<)')

    for tg_path in tqdm.tqdm(textgrid_files, desc='TextGrid'):
        session, clip_id, _ = parse_wav_meta(tg_path.replace('.TextGrid', '.wav'), data_dir)
        tg = tool.read_textgrid_from_file(tg_path)
        content_tier = tg.tiers[0]
        for interval in content_tier.intervals:
            text = interval.content.strip()
            if not text or text in NOISE_MARKERS or skip_pattern.match(text):
                continue
            if not re.search(r'[\u4e00-\u9fa5]', text):
                continue
            session_lines.setdefault(session, []).append(
                (round(interval.xmin, 2), round(interval.xmax, 2), clip_id, text)
            )

    meeting_dict = {
        session: build_meeting_text(lines)
        for session, lines in session_lines.items()
    }
    reference_json = os.path.join(output_dir, 'meeting_reference.json')
    tool.json2dic(reference_json, meeting_dict)
    print('Saved reference summaries input:', reference_json)
    return reference_json


def parse_args():
    parser = argparse.ArgumentParser(description='WhisperX recognition for dev-F8N')
    parser.add_argument('--data_dir', default=DEFAULT_DATA_DIR)
    parser.add_argument('--output_dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--model', default='small', help='WhisperX model size (small/base/large-v3)')
    parser.add_argument('--language', default='zh')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--sessions', nargs='*', help='Optional session ids, e.g. M014 M016')
    parser.add_argument('--build_reference', action='store_true', help='Export TextGrid reference meetings')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.build_reference:
        build_reference_from_textgrid(args.data_dir, args.output_dir)
    run_recognition(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        language=args.language,
        batch_size=args.batch_size,
        device=args.device,
        sessions=args.sessions,
    )
