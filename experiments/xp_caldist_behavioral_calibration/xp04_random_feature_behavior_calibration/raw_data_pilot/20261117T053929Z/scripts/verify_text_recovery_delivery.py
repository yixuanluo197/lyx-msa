"""Verify blocked audit evidence without source-pickle/model loading or inference."""
from pathlib import Path
import ast,csv,datetime,hashlib,io,json,re,subprocess,zipfile
OUT=Path(__file__).resolve().parents[1]
WORK=Path('/LOCAL/lyx-msa')

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(path):return json.loads(Path(path).read_text())
class TailReader(io.RawIOBase):
 def __init__(self,tail,total):self.data=tail;self.total=total;self.start=total-len(tail);self.position=0;self.bytes_served=0
 def seek(self,offset,whence=0):
  self.position=offset+(0 if whence==0 else self.position if whence==1 else self.total);return self.position
 def tell(self):return self.position
 def seekable(self):return True
 def read(self,size=-1):
  if size<0:size=self.total-self.position
  end=min(self.position+size,self.total)
  if self.position<self.start or end>self.total:raise ValueError('Refuse any ZIP payload read outside cached tail')
  value=self.data[self.position-self.start:end-self.start];self.position=end;self.bytes_served+=len(value);return value

def main():
 checks={};anchors=load(OUT/'provenance/input_hashes_before.json');history=load(OUT/'provenance/history_hashes_before.json')
 for name,digest in anchors.items():checks['original_unchanged:'+name]=sha(name)==digest
 for name,digest in history.items():checks['history_unchanged:'+name]=sha(OUT.parent/name)==digest
 idx=load(OUT/'provenance/input_index_audit.json');status=load(OUT/'provenance/text_parity_status.json');rows=list(csv.DictReader((OUT/'layer_comparison.csv').open()))
 checks['eight_layer_status_rows']=len(rows)==8
 for ds,n,g,slots in [('IEMOCAP',527,12,[2,3,4,5]),('MELD',1109,114,[3,4,5,6])]:
  r=idx['datasets'][ds];s=status[ds]
  checks[ds+':effective_manifest_boundary']=r['validation_rows']==n and r['validation_dialogues']==g
  checks[ds+':weights_not_substituted']=s['actual_cmerc_encoder_found'] is False and s['successful_extraction'] is False
  checks[ds+':not_run_not_false_numeric_failure']=s['all_four_clean_parity_passed'] is None and s['numeric_feature_failure_observed'] is False
  checks[ds+':no_probability_forward']=s['probability_status']=='NOT_RUN' and s['probability_gate_eligible'] is False and all(v is None for v in s['probability_metrics'].values())
  checks[ds+':unmeasured_all_layers']=all(r['status']=='NOT_RUN_MISSING_CONFIRMED_ENCODER' and r['extracted_shape'] is None and r['MAE'] is None and r['passed'] is None for r in s['layers'])
  checks[ds+':reference_slot_order']= [r['reference_pickle_slot'] for r in s['layers']]==slots and [r['reference_hidden_offset'] for r in s['layers']]==[-1,-2,-3,-4]
  checks[ds+':CSV_all_NA']=all(r['MAE']=='NA' and r['passed']=='NA' for r in rows if r['dataset']==ds)
 paper=load(OUT/'provenance/paper_source_evidence.json')
 checks['paper_does_not_authorize_unfinetuned_substitution']=paper['decision']['paper_explicitly_authorizes_untuned_public_roberta_as_actual_extractor'] is False
 checks['formal_CMERC_paper_hash_matches']=sha(paper['local_final_paper']['path'])==paper['local_final_paper']['sha256']
 for rec in paper['sources']:checks['paper_source_hash:'+Path(rec['path']).name]=sha(rec['path'])==rec['sha256']
 cached=load(OUT/'provenance/reference_source_cache_resolution.json')
 for rec in cached['records']:checks['cached_fixed_reference:'+rec['file']]=sha(OUT/'sources/verified_cached_reference'/rec['file'])==rec['sha256']
 frozen=load(OUT/'provenance/protocol_frozen.json');checks['protocol_unchanged_and_no_Val_access']=sha(OUT/'configs/TEXT_RECOVERY_PROTOCOL.json')==frozen['protocol_sha256'] and frozen['validation_values_opened'] is False
 listing=load(OUT/'provenance/cosmic_archive_directory.json');tail=(OUT/'local_only/cosmic_archive_tail.bin').read_bytes();reader=TailReader(tail,listing['remote_archive_bytes'])
 with zipfile.ZipFile(reader) as archive:
  independent=[{'name':r.filename,'uncompressed_bytes':r.file_size,'compressed_bytes':r.compress_size} for r in archive.infolist()]
 expected=[{k:r[k] for k in ('name','uncompressed_bytes','compressed_bytes')} for r in listing['entries']]
 checks['independent_zipfile_directory_matches_manual_parser']=independent==expected
 checks['archive_only_metadata_no_decompression']=listing['directory_fully_listed'] is True and listing['payloads_decompressed']==0 and len(tail)==131072
 checks['archive_no_weight_named_entries']=listing['model_weight_named_entries']==[]
 comparison=load(OUT/'provenance/cosmic_cmerc_zip_metadata_comparison.json')
 checks['both_feature_headers_match_size_CRC_only']=all(r['header_size_and_CRC32_match'] is True and r['COSMIC_payload_SHA256_measured'] is False for r in comparison['comparisons'])
 required=['ROBERTA_SOURCE_RECOVERY.md','IEMOCAP_TEXT_CLEAN_PARITY.md','MELD_TEXT_CLEAN_PARITY.md','XP04_TEXT_PROBABILITY_PARITY.md','FINAL_TEXT_RECOVERY_DECISION.md','README.md']
 for name in required:checks['report_exists:'+name]=(OUT/name).is_file()
 for path in (OUT/'scripts').glob('*.py'):ast.parse(path.read_text());checks['python_syntax:'+path.name]=True
 for path in OUT.glob('*.md'):
  for link in re.findall(r'\]\(([^)]+)\)',path.read_text()):
   if '://' in link or link.startswith('#') or link=='verification.json':continue
   checks['link:'+path.name+':'+link]=(path.parent/link.split('#',1)[0]).exists()
 head=subprocess.check_output(['git','-C',str(WORK/'CMERC'),'rev-parse','HEAD'],text=True).strip();checks['Git_HEAD_unchanged_no_commit']=head==idx['git_head']
 checks['no_staged_changes']=not subprocess.check_output(['git','-C',str(WORK/'CMERC'),'diff','--cached','--name-only'],text=True).strip()
 checks['no_tracked_CMERC_changes']=not subprocess.check_output(['git','-C',str(WORK/'CMERC'),'diff','--name-only'],text=True).strip()
 checks['private_dir_0700']=(OUT/'local_only').stat().st_mode&0o777==0o700
 review=OUT/'provenance/independent_text_delivery_review.json'
 if review.exists():checks['independent_delivery_review_approved']=load(review)['status']=='APPROVED'
 hashes={str(f.relative_to(OUT)):sha(f) for f in OUT.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.name not in ('verification.json','artifact_hashes_final.json')}
 manifest=OUT/'provenance/artifact_hashes_final.json';manifest.write_text(json.dumps(hashes,indent=2)+'\n')
 result={'status':'PASS' if all(checks.values()) else 'FAIL','meaning':'Audit delivery verification, not feature parity success','scientific_status':'BLOCKED_MISSING_CONFIRMED_ENCODER','verified_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'check_count':len(checks),'failed_checks':[k for k,v in checks.items() if not v],'checks':checks,'frozen_inputs_checked':len(anchors),'historical_files_checked':len(history),'artifacts_hashed':len(hashes),'artifact_manifest_sha256':sha(manifest),'datasets':status,'execution':{'text_encoder_loads':0,'raw_text_extraction_runs':0,'validation_target_value_access':0,'CMERC_checkpoint_loads':0,'CMERC_forward_runs':0,'training_runs':0,'affine_or_regression_converters':0,'audio_visual_processing':0,'test_label_access':0,'perturbation_runs':0,'git_commit_push':False,'full_model_archive_downloads':0,'COSMIC_remote_range_bytes':len(tail),'COSMIC_zip_payloads_decompressed':0},'repository_head':head,'known_failures_retained':['initial ZIP directory count assertion','source manifest mixed URL keys','raw source proxy502','root/feature README filename collision','ACM403/OpenReview browser verification/Drive SSL timeout/Dropbox502'],'limits':'No actual producer checkpoint identity established. Expected reference shapes are inherited from anchored historical structure audit, not newly measured target tensors. Frozen thresholds exist but no numeric parity or probabilities were measured.'}
 (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'status':result['status'],'checks':len(checks),'failed_checks':result['failed_checks'],'artifacts_hashed':len(hashes),'scientific_status':result['scientific_status']}))
 if result['status']!='PASS':raise SystemExit(1)
if __name__=='__main__':main()
