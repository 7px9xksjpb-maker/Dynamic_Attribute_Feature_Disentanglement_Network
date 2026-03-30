import codecs
import re

# 1. Generate udafd_sparse_attn.py
with codecs.open("d:/HRRP/A02/DA_Feature/udafd_compress128.py", "r", "utf-8") as f:
    text = f.read()

start_idx = text.find("class RangeCompressor(nn.Module):")
end_idx = text.find("def maybe_compress_sequence(")

text = text[:start_idx] + "from SparseAttention import SparseAttentionCompressor\n\n\n" + text[end_idx:]

text = text.replace("RangeCompressor", "SparseAttentionCompressor")
text = text.replace("use_range_compressor: bool = True", "use_sparse_attention: bool = True")
text = text.replace("cfg.use_range_compressor", "cfg.use_sparse_attention")
text = text.replace("compressor_hidden_channels", "attention_hidden_channels")

# remove kernel_size and use_skip from config definition
text = re.sub(r'    compressor_kernel_size: int = 5\n\s*compressor_use_skip: bool = True\n', '', text)

# fix instantiation
text = re.sub(
    r'range_compressor\s*=\s*SparseAttentionCompressor\(\s*input_bins=raw_input_dim,\s*output_bins=cfg\.compressed_bins,\s*hidden_channels=cfg\.attention_hidden_channels,\s*kernel_size=cfg\.compressor_kernel_size,\s*use_skip=cfg\.compressor_use_skip,\s*\)',
    r'range_compressor = SparseAttentionCompressor(\n            input_bins=raw_input_dim,\n            output_bins=cfg.compressed_bins,\n            hidden_channels=cfg.attention_hidden_channels,\n        )',
    text
)

# replace import from old file
text = text.replace("'RangeCompressor'", "'SparseAttentionCompressor'")

with codecs.open("d:/HRRP/A02/DA_Feature/udafd_sparse_attn.py", "w", "utf-8") as f:
    f.write(text)


# 2. Generate train_udafd_sparse_attn.py
with codecs.open("d:/HRRP/A02/DA_Feature/train_udafd_compress.py", "r", "utf-8") as f:
    train_text = f.read()

train_text = train_text.replace("udafd_compress128", "udafd_sparse_attn")
train_text = train_text.replace("outputs3_compress", "outputs3_sparse_attn")

train_text = train_text.replace("USE_RANGE_COMPRESSOR", "USE_SPARSE_ATTENTION")
train_text = train_text.replace("COMPRESSOR_HIDDEN_CHANNELS", "ATTENTION_HIDDEN_CHANNELS")

# Remove kernel_size and skip
train_text = re.sub(r'COMPRESSOR_KERNEL_SIZE = 5\nCOMPRESSOR_USE_SKIP = True\n', '', train_text)

train_text = train_text.replace("use_range_compressor=USE_SPARSE_ATTENTION", "use_sparse_attention=USE_SPARSE_ATTENTION")
train_text = train_text.replace("compressor_hidden_channels=ATTENTION_HIDDEN_CHANNELS", "attention_hidden_channels=ATTENTION_HIDDEN_CHANNELS")

# Fix kwargs
train_text = re.sub(r'        compressor_kernel_size=COMPRESSOR_KERNEL_SIZE,\n\s*compressor_use_skip=COMPRESSOR_USE_SKIP,\n', '', train_text)

with codecs.open("d:/HRRP/A02/DA_Feature/train_udafd_sparse_attn.py", "w", "utf-8") as f:
    f.write(train_text)


# 3. Generate test_udafd_sparse_attn.py
with codecs.open("d:/HRRP/A02/DA_Feature/test_udafd_compress.py", "r", "utf-8") as f:
    test_text = f.read()

test_text = test_text.replace("udafd_compress128", "udafd_sparse_attn")
test_text = test_text.replace("outputs3_compress", "outputs3_sparse_attn")

test_text = test_text.replace("RangeCompressor", "SparseAttentionCompressor")

# Replace config access
test_text = test_text.replace("getattr(cfg, 'use_range_compressor'", "getattr(cfg, 'use_sparse_attention'")
test_text = test_text.replace("cfg.compressor_hidden_channels", "cfg.attention_hidden_channels")

# Fix test instantiation
test_text = re.sub(
    r'range_compressor = SparseAttentionCompressor\(\s*input_bins=raw_input_dim,\s*output_bins=cfg\.compressed_bins,\s*hidden_channels=cfg\.attention_hidden_channels,\s*kernel_size=cfg\.compressor_kernel_size,\s*use_skip=cfg\.compressor_use_skip,\s*\)',
    r'range_compressor = SparseAttentionCompressor(\n            input_bins=raw_input_dim,\n            output_bins=cfg.compressed_bins,\n            hidden_channels=cfg.attention_hidden_channels,\n        )',
    test_text
)

with codecs.open("d:/HRRP/A02/DA_Feature/test_udafd_sparse_attn.py", "w", "utf-8") as f:
    f.write(test_text)

print("success")
