import json,urllib.request,hashlib,concurrent.futures,pathlib
BASE=pathlib.Path(__file__).parent
OUT=BASE/'image_paths'
OUT.mkdir(exist_ok=True)
manifest=json.loads((OUT/'download_manifest.json').read_text(encoding='utf-8-sig'))
def fetch(r):
 r=dict(r)
 try:
  req=urllib.request.Request(r['url'],headers={'User-Agent':'Mozilla/5.0'})
  with urllib.request.urlopen(req,timeout=45) as response:
   data=response.read(25000000)
  if data[:3]==b'\xff\xd8\xff': ext='.jpg'
  elif data[:8]==b'\x89PNG\r\n\x1a\n': ext='.png'
  elif data[:4]==b'RIFF' and data[8:12]==b'WEBP': ext='.webp'
  elif data[:6] in (b'GIF87a',b'GIF89a'): ext='.gif'
  else: raise ValueError('Response is not a recognized image')
  name=r['sample_id']+('_source' if r['kind']=='source_screenshot' else '')+ext
  path=OUT/name
  if path.exists() and path.read_bytes()!=data: raise ValueError('Existing different file; preserved')
  path.write_bytes(data)
  r.update(status='downloaded',filename=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
  try:
   from PIL import Image
   with Image.open(path) as im:
    r['dimensions']=list(im.size); im.verify()
   r['decode_verified']=True
  except ImportError: r['decode_verified']=False
 except Exception as e: r.update(status='failed',error=str(e))
 print(r['sample_id'],r['status'],r.get('error',''),flush=True)
 return r
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool: results=list(pool.map(fetch,manifest))
(OUT/'download_report.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print('DONE',sum(r['status']=='downloaded' for r in results),'/',len(results),flush=True)
