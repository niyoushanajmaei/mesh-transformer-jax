#use the inf-env to run this.
#pip install -r mesh-transformer-jax/requirements.txt
#pip install mesh-transformer-jax/ jax==0.2.12 tensorflow==2.5.0
#infering with TPU shards written by TPU

import time
import jax
from jax.experimental import maps
import numpy as np
import optax
import transformers
from pathlib import Path
import os

from mesh_transformer.checkpoint import read_ckpt
from mesh_transformer.sampling import nucleaus_sample
from mesh_transformer.transformer_shard import CausalTransformer

params = {
  "layers": 28,
  "d_model": 4096,
  "n_heads": 16,
  "n_vocab": 50400,
  "norm": "layernorm",
  "pe": "rotary",
  "pe_rotary_dims": 64,

  "seq": 2048,
  "cores_per_replica": 1,
  "per_replica_batch": 1,
}

per_replica_batch = params["per_replica_batch"]
cores_per_replica = params["cores_per_replica"]
seq = params["seq"]


params["sampler"] = nucleaus_sample

params["optimizer"] = optax.scale(0)

mesh_shape = (jax.device_count() // cores_per_replica, cores_per_replica)
devices = np.array(jax.devices()).reshape(mesh_shape)

maps.thread_resources.env = maps.ResourceEnv(maps.Mesh(devices, ('dp', 'mp')))

tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')

total_batch = per_replica_batch * jax.device_count() // cores_per_replica

network = CausalTransformer(params)

network.state = read_ckpt(network.state, "/home/zero11/step_383500/", devices.shape[1])

network.state = network.move_xmap(network.state, np.zeros(cores_per_replica))

def infer(context, top_p=0.9, temp=1.0, gen_len=512):
    tokens = tokenizer.encode(context)

    provided_ctx = len(tokens)
    pad_amount = seq - provided_ctx

    padded_tokens = np.pad(tokens, ((pad_amount, 0),)).astype(np.uint32)
    batched_tokens = np.array([padded_tokens] * total_batch)
    length = np.ones(total_batch, dtype=np.uint32) * len(tokens)

    start = time.time()
    output = network.generate(batched_tokens, length, gen_len, {"top_p": np.ones(total_batch) * top_p, "temp": np.ones(total_batch) * temp})

    samples = []
    decoded_tokens = output[1][0]

    for o in decoded_tokens[:, :, 0]:
      samples.append(f"\033[1m{context}\033[0m{tokenizer.decode(o)}")

    print(f"completion done in {time.time() - start:06}s")
    return samples

def infer_test_set():
  p =[]
  d = "test_set"
  c=0
  for path in os.listdir(d):
      full_path = os.path.join(d, path)
      c+=1
      if os.path.isfile(full_path):
          txt = Path(full_path).read_text()
          txt = txt.replace('\n', '')
          result = infer(txt)
          with open("infer/results/"+'product'+c+".txt", 'w') as writer:
            writer.write(result)

print(infer("EleutherAI is")[0])
