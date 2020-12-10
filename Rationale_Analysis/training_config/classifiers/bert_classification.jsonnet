local berts = import '../berts.libsonnet'; 

{
  dataset_reader : {
    type : "base_reader",
    token_indexers : {
      bert : berts.indexer,
    },
    cache_directory : std.extVar("DATA_BASE_PATH"),
  },
  validation_dataset_reader: {
    type : "base_reader",
    token_indexers : {
      bert : berts.indexer,
    },
    cache_directory : std.extVar("DATA_BASE_PATH"),
  },
  train_data_path: std.extVar('TRAIN_DATA_PATH'),
  validation_data_path: std.extVar('DEV_DATA_PATH'),
  test_data_path: std.extVar('TEST_DATA_PATH'),
  model: berts.classifier,
  data_loader : {
    shuffle: true,
    samplers: {
        type: "max_tokens_batch_sampler",
        max_tokens: std.parseInt(std.extVar('BSIZE')),
    }
  },
  trainer: {
    num_epochs: std.parseInt(std.extVar('EPOCHS')),
    patience: 10,
    grad_norm: 5.0,
    validation_metric: "+validation_metric",
    checkpointer: {num_serialized_models_to_keep: 1,},
    cuda_device: std.parseInt(std.extVar("CUDA_DEVICE")),
    optimizer: {
      type: "adamw",
      lr: 2e-5
    }
  },
  random_seed:  std.parseInt(std.extVar("SEED")),
  pytorch_seed: std.parseInt(std.extVar("SEED")),
  numpy_seed: std.parseInt(std.extVar("SEED")),
  evaluate_on_test: false
}
