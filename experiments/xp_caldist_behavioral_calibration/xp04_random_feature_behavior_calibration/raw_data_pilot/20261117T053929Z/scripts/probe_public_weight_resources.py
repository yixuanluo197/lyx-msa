"""Bounded metadata/range probes only; no model or full archive download."""
from pathlib import Path
import datetime,json,urllib.request
OUT=Path(__file__).resolve().parents[1]
checks=[]
items=[('fairseq_pretrained','https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz','HEAD',{}),('cosmic_feature_archive_tail','https://drive.usercontent.google.com/download?id=1TQYQYCoPtdXN2rQ1mR2jisjUztmOzfZr&export=download&confirm=t','GET',{'Range':'bytes=-131072'})]
for name,url,method,headers in items:
 row={'name':name,'url':url,'method':method,'requested_range':headers.get('Range'),'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 try:
  request=urllib.request.Request(url,method=method,headers={'User-Agent':'Mozilla/5.0',**headers})
  with urllib.request.urlopen(request,timeout=15) as response:
   row.update(status=response.status,content_type=response.headers.get('Content-Type'),content_length=response.headers.get('Content-Length'),content_range=response.headers.get('Content-Range'))
   if method=='GET' and response.status==206:
    tail=response.read(131073);row['bytes_received']=len(tail)
    if len(tail)>131072:raise ValueError('Range larger than permitted probe')
    (OUT/'local_only/cosmic_archive_tail.bin').write_bytes(tail)
    row.update(zip_end_record_found=b'PK\x05\x06' in tail,archive_directory_parsed=False)
   elif method=='GET':row.update(bytes_received=0,body_not_read='Server did not honor bounded range; no full download')
   else:row['bytes_received']=0
 except Exception as error:row.update(error=type(error).__name__,message=str(error))
 checks.append(row)
result={'checks':checks,'large_weight_downloads':0,'zip_unpacked':False,'encoder_identity_established':False,'archive_absence_claimed':False}
with (OUT/'provenance/public_resource_probe.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
