#infering with CPU the shards written by TPU. 
#pip install -r requirements.txt
#pip install jax==0.2.12 tensorflow==2.5.0

import time

import jax
from jax.experimental import maps
import numpy as np
import optax
import transformers
import resource as re
import time

from mesh_transformer.checkpoint import read_ckpt
from mesh_transformer.sampling import nucleaus_sample
from mesh_transformer.transformer_shard import CausalTransformer
from mesh_transformer import util
from mesh_transformer.util import clip_by_global_norm, additive_weight_decay

params = {
  "layers": 28,
  "d_model": 4096,
  "n_heads": 16,
  "n_vocab": 50400,
  "norm": "layernorm",
  "pe": "rotary",
  "pe_rotary_dims": 64,
  "gradient_accumulation_steps": 16,
  "warmup_steps": 7,
  "anneal_steps": 65,
  "lr": 5e-5,
  "end_lr": 1e-5,
  "weight_decay": 0.1,
  "total_steps": 72,
  "early_cast": True,
  "seq": 2048,
  "cores_per_replica": 1, 
  "per_replica_batch": 1,
}

per_replica_batch = params["per_replica_batch"]
cores_per_replica = params["cores_per_replica"]
seq = params["seq"]

params["sampler"] = nucleaus_sample

# optimizer for the slim weights
# params["optimizer"] = optax.scale(0)

#optimizer for the full weights
#gradient_accumulation_steps = params.get("gradient_accumulation_steps", 1)
#weight_decay = params["weight_decay"]
#warmup_steps = params["warmup_steps"]
#anneal_steps = params["anneal_steps"]
#end_lr = params["end_lr"]
#lr = params["lr"]
#scheduler = util.gpt3_schedule(warmup_steps, anneal_steps, lr, end_lr)
#opt = optax.chain(
#        optax.scale(1 / gradient_accumulation_steps),
#        clip_by_global_norm(1),
#        optax.scale_by_adam(),
#        additive_weight_decay(weight_decay),
#        optax.scale(-1),
#        optax.scale_by_schedule(scheduler)
#    )
#params["optimizer"] = opt

devices = np.array([jax.devices()[0]]).reshape((1, 1))
maps.thread_resources.env = maps.ResourceEnv(maps.Mesh(devices, ('dp', 'mp')))

tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')

network = CausalTransformer(params)

start = time.time()

# here we load a checkpoint which was written with 8 shards into 1 shard
network.state = read_ckpt(network.state, "/home/zero11/slim/step_383500/", 8, shards_out=cores_per_replica)

print(f"loading RAM usage: {re.getrusage(re.RUSAGE_SELF)}")

# move the state to CPU/system memory so it's not duplicated by xmap
network.state = jax.device_put(network.state, jax.devices("cpu")[0])

def infer(context, top_k=40, top_p=0.9, temp=1.0, gen_len=210):
    tokens = tokenizer.encode(context)

    provided_ctx = len(tokens)
    pad_amount = seq - provided_ctx

    padded_tokens = np.pad(tokens, ((pad_amount, 0),)).astype(np.uint32)
    batched_tokens = np.array([padded_tokens] * per_replica_batch)
    length = np.ones(per_replica_batch, dtype=np.uint32) * len(tokens)

    start = time.time()
    output = network.generate(batched_tokens, length, gen_len, {"top_p": np.ones(per_replica_batch) * top_p, "top_k": top_k is not None and (np.ones(per_replica_batch, dtype=np.int32) * top_k) or None, "temp": np.ones(per_replica_batch) * temp})

    samples = []
    decoded_tokens = output[1][0]

    for o in decoded_tokens[:, :, 0]:
      samples.append(tokenizer.decode(o))

    print(f"total RAM usage: {re.getrusage(re.RUSAGE_SELF)}")
    print(f"completion done in {time.time() - start:06}s")
    return samples


print(infer("EleutherAI is"))
