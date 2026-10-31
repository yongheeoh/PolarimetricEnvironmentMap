"""Small lossless float OpenEXR scanline writer (ZIP, arbitrary named channels).

Uses the public EXR scanline/ZIP layout; readable by the desktop OIIO/polanalyser
readers. Compression changes storage only, never the floating-point samples.
"""
import struct
import zlib
from pathlib import Path
import numpy as np


def write_exr(path, channels, check=lambda:None):
    names=sorted(channels)
    height,width=channels[names[0]].shape
    if any(a.shape!=(height,width) for a in channels.values()):raise ValueError('EXR channel shape mismatch')
    def attr(name,kind,value):return name.encode()+b'\0'+kind.encode()+b'\0'+struct.pack('<I',len(value))+value
    ch=b''.join(n.encode()+b'\0'+struct.pack('<iB3xii',2,0,1,1) for n in names)+b'\0'
    box=struct.pack('<4i',0,0,width-1,height-1)
    header=struct.pack('<II',20000630,2)+attr('channels','chlist',ch)
    header+=attr('compression','compression',b'\x03')+attr('dataWindow','box2i',box)+attr('displayWindow','box2i',box)
    header+=attr('lineOrder','lineOrder',b'\0')+attr('pixelAspectRatio','float',struct.pack('<f',1))
    header+=attr('screenWindowCenter','v2f',struct.pack('<2f',0,0))+attr('screenWindowWidth','float',struct.pack('<f',1))+b'\0'
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    chunks=(height+15)//16
    with path.open('wb') as f:
        f.write(header);table=f.tell();f.write(b'\0'*(chunks*8));offsets=[]
        for y in range(0,height,16):
            check()
            rows=[]
            for row in range(y,min(y+16,height)):
                for name in names:
                    values=np.asarray(channels[name][row],dtype='<f4')
                    if not np.isfinite(values).all():raise ValueError('Nonfinite EXR')
                    rows.append(values.tobytes())
            raw=b''.join(rows);bytes_=np.frombuffer(raw,np.uint8)
            split=np.r_[bytes_[::2],bytes_[1::2]]
            predicted=split.copy();predicted[1:]=(split[1:].astype(np.int16)-split[:-1]+128).astype(np.uint8)
            compressed=zlib.compress(predicted.tobytes(),4)
            body=compressed if len(compressed)<len(raw) else raw
            offsets.append(f.tell());f.write(struct.pack('<ii',y,len(body)));f.write(body)
        f.seek(table);f.write(struct.pack('<'+'Q'*len(offsets),*offsets))


def payload_channels(data, prefix='S'):
    channels={color:data[...,c,0] for c,color in enumerate('BGR')}
    if data.shape[-1]>1:
        for i in range(data.shape[-1]):
            key=f'S2L2{i:02d}' if prefix=='S2L2' else f'S{i}'
            channels.update({f'{key}.{color}':data[...,c,i] for c,color in enumerate('BGR')})
    return channels
