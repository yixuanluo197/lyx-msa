"""Create a public history reference; preserve the original frozen bytes locally."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'provenance/xp04_validation_reference.json'
    fields = []

    def redact(value, prefix=''):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                path = prefix + '.' + key
                if key.endswith('_ids') and isinstance(item, list):
                    fields.append({'field': path, 'removed_count': len(item)})
                    result[key + '_count'] = len(item)
                else:
                    result[key] = redact(item, path)
            return result
        if isinstance(value, list):
            return [redact(item, prefix + '[]') for item in value]
        return value

    result = redact(json.loads(source.read_text()))
    result['public_redaction'] = {
        'original_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'removed_id_fields': fields,
        'original_preserved_server_only': True,
        'scientific_configuration_and_metrics_changed': False,
    }
    target = ROOT / 'provenance/xp04_validation_reference_public.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'public_copy': str(target), 'removed_fields': len(fields)}))


if __name__ == '__main__':
    main()
