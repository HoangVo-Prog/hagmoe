# Loss Params Helper

Canonical class weights are defined in `scripts/_loss_params.sh` in the order:

pos, neg, neu

At runtime, each script:
- loads dataset-specific base weights and focal gamma
- reads `label2id` via `Config.from_cli().finalize().validate()` and `get_dataset`
- reorders the weights to match numeric label ids

Verify the mapping by checking the printed line:

LABEL_ORDER: id0=<label> id1=<label> id2=<label> -> CLASS_WEIGHTS=<...>
