"""Parse only cached ZIP central-directory metadata; never decompress file payloads."""
from pathlib import Path
import hashlib,json,struct
OUT=Path(__file__).resolve().parents[1]

def main():
 probe=json.loads((OUT/'provenance/public_resource_probe.json').read_text())['checks'][1]
 assert probe['status']==206
 total=int(probe['content_range'].rsplit('/',1)[1]);start=int(probe['content_range'].split()[1].split('-')[0])
 path=OUT/'local_only/cosmic_archive_tail.bin';tail=path.read_bytes()
 pos=tail.rfind(b'PK\x05\x06');end=struct.unpack_from('<4s4H2LH',tail,pos)
 count,size,offset=end[4],end[5],end[6]
 if offset==0xffffffff:
  z=tail.rfind(b'PK\x06\x06');assert z>=0
  record=struct.unpack_from('<4sQ2H2L4Q',tail,z);count,size,offset=record[7],record[8],record[9]
 assert start<=offset and offset+size<=start+len(tail)
 directory=tail[offset-start:offset-start+size];cursor=0;entries=[]
 for _ in range(count):
  values=struct.unpack_from('<4s6H3L5H2L',directory,cursor);assert values[0]==b'PK\x01\x02'
  n,extra,comment=values[10:13];name=directory[cursor+46:cursor+46+n].decode('utf-8' if values[3]&0x800 else 'cp437')
  entries.append({'name':name,'uncompressed_bytes':values[9],'compressed_bytes':values[8],'compression_method':values[4],'encrypted':bool(values[3]&1)})
  cursor+=46+n+extra+comment
 assert cursor==len(directory)
 candidates=[r['name'] for r in entries if Path(r['name']).suffix in ('.pt','.pth','.bin','.safetensors','.hdf5') or 'checkpoint' in r['name'].lower()]
 result={'source_url':probe['url'],'official_readme_name':'erc-training.zip','remote_archive_bytes':total,'range_bytes_read':len(tail),'central_directory_bytes':size,'entries_count':len(entries),'entries':entries,'model_weight_named_entries':candidates,'directory_fully_listed':True,'payloads_decompressed':0,'models_or_datasets_loaded':False,'tail_sha256':hashlib.sha256(tail).hexdigest(),'limit':'Names and ZIP headers only. Pickle contents and any nested semantics were not inspected; no global claim of absence of hidden encoder bytes.'}
 with (OUT/'provenance/cosmic_archive_directory.json').open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
