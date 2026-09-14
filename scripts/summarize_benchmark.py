"""Combine recorded benchmark measurements and verify the delivered artifacts."""
from pathlib import Path
import argparse
import hashlib
import json
import statistics

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, help='Optional earlier final-image SHA256 JSON for exact regression comparison')
    args = parser.parse_args()
    models = []
    for path in sorted((ROOT / 'outputs/segmentation').glob('*/metrics.json')):
        measured = json.loads(path.read_text())
        quality_path = path.parent / 'quality.json'
        quality = json.loads(quality_path.read_text()) if quality_path.exists() else {}
        models.append({
            'run': path.parent.name,
            'model': measured['model'],
            'mean_warmed_inference_seconds': measured['mean_median_wall_seconds'],
            'peak_rss_gib': measured['process_peak_rss_mib'] / 1024,
            'reference_proxy_iou': quality.get('mean_iou'),
            'reference_proxy_boundary_f1_2px': quality.get('mean_boundary_f1_2px'),
        })
    pipeline = json.loads((ROOT / 'outputs/phase1/metrics.json').read_text())
    robustness = json.loads((ROOT / 'outputs/robustness/metrics.json').read_text())['images']
    expected = json.loads(args.baseline.read_text()) if args.baseline else {}
    validated = []
    for row in pipeline['images']:
        name = Path(row['image']).stem
        directory = ROOT / 'outputs/phase1' / name
        required = [
            'final_white.png', 'segmentation_mask.png', 'refined_mask.png',
            'geometry_contacts.png', 'normalized_geometry.png',
            'normalized_placement.png', 'normalized_alpha.png',
            'shadow_tire_contact_alpha.png', 'shadow_underbody_alpha.png',
            'shadow_ground_alpha.png', 'shadow_combined_alpha.png',
            'comparison.jpg', 'reference_aligned.png', 'metadata.json',
        ]
        assert all((directory / file).exists() for file in required), name
        assert Image.open(directory / 'final_white.png').size == (1024,768), name
        digest = hashlib.sha256((directory / 'final_white.png').read_bytes()).hexdigest()
        if expected:
            assert digest == expected[name], f'Original result changed: {name}'
        assert row['segmentation_info']['inference_passes'] == 1, name
        validated.append(name)
    assert len(validated) == 8
    assert len(robustness) == 16
    stage_names = ['mask_refinement_seconds', 'geometry_seconds', 'placement_seconds',
                   'shadow_and_compositing_seconds']
    summary = {
        'selected_model': 'birefnet-general',
        'selection_reason': 'Best inspected edge/mirror preservation and highest reference-proxy mask agreement; CPU quality takes priority over speed.',
        'models': models,
        'pipeline': {
            'mean_warm_seconds': statistics.mean(row['timings']['processing_seconds'] for row in pipeline['images'][1:]),
            'first_image_seconds_including_load': pipeline['images'][0]['timings']['processing_seconds'],
            'peak_rss_gib': pipeline['process_peak_rss_mib'] / 1024,
            'mean_postprocessing_seconds': statistics.mean(sum(row['timings'][key] for key in stage_names) for row in pipeline['images']),
            'timing_note': pipeline['timing_note'],
        },
        'robustness': {
            'count': len(robustness),
            'mean_mask_consistency_iou': statistics.mean(row['mask_consistency_iou'] for row in robustness),
            'minimum_mask_consistency_iou': min(row['mask_consistency_iou'] for row in robustness),
            'maximum_normalized_bbox_delta_pixels': max(row['normalized_bbox_max_delta_pixels'] for row in robustness),
        },
        'artifact_validation': {
            'views': validated,
            'all_required_stages_present': True,
            'all_final_images_byte_identical_to_supplied_baseline': True if expected else None,
            'all_original_images_use_single_inference': True,
        },
        'limitations': [
            'Eight views of one sedan; controlled variants are not unseen vehicles.',
            'Reference-derived masks are approximate and do not score shadow realism.',
            'Geometry is heuristic; hidden contacts and track are inferred.',
            'Small subjects may require an extra inference, increasing runtime.',
        ],
    }
    (ROOT / 'outputs/benchmark_summary.json').write_text(json.dumps(summary,indent=2) + '\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
