import sys
import os
parent_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)),'..')
sys.path.insert(1, parent_dir)
sys.path.insert(1, os.path.join(parent_dir,'graphnn'))

from graphnn import data
import argparse
import torch
import PAINN_var
import json
import os
import logging
import PAINN_var
import DBTransformer

def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Evaluate graph convolution network", fromfile_prefix_chars="+"
    )
    parser.add_argument(
        "--load_model",
        type=str,
        default=None,
        help="Load specified variational PAINN model",
    )
    parser.add_argument(
        "--load_model_args",
        type=str,
        default=None,
        help="Load specified variational PAINN model arguments",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Load specified dataset",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Set which device to use for training e.g. 'cuda' or 'cpu'",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tests/model_evaluation",
        help="Path to output directory",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default=None,
        help="Train/test/validation split file json",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=100,
        help="Number of samples per molecule",
    )
    parser.add_argument(
        "--forces_property",
        type=str,
        default="forces",
        help="Name of forces property in ASE database",
    )
    parser.add_argument(
        "--U0_to_E",
        action="store_true",
        help="Transform U0 to E by subtracting zpve",
    )
    parser.add_argument(
        "--compute_forces",
        action="store_true",
        help="Compute forces",
    )

    return parser.parse_args(arg_list)

if __name__ == "__main__":
    args = get_arguments()

    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(args.output_dir, "printlog.txt"), mode="w"
            ),
            logging.StreamHandler(),
        ],
    )

    logging.info("Starting evaluation.")
    
    # Create device
    device = torch.device(args.device)
    state_dict = torch.load(args.load_model)
    with open(args.load_model_args, "r") as af:
        model_args = json.load(af)
    
    # Assumes default value if key is missing
    if model_args is None:
        logging.info("No model arguments provided. Assuming all default values.")
        model_args = {}

    if "hidden_state_size" not in model_args:
        logging.info("No hidden_state_size provided. Assuming default value.")
        model_args["hidden_state_size"] = 64 
    
    if "num_interactions" not in model_args:
        logging.info("No num_interactions provided. Assuming default value.")
        model_args["num_interactions"] = 3
    
    if "cutoff" not in model_args:
        logging.info("No cutoff provided. Assuming default value.")
        model_args["cutoff"] = 5.0
    
    if "atomwise_normalization" not in model_args:
        logging.info("No atomwise_normalization provided. Assuming default value.")
        model_args["atomwise_normalization"] = False

    if "direct_force_output" not in model_args:
        logging.info("No direct_force_output provided. Assuming default value.")
        model_args["direct_force_output"] = False

    if "standard_deviance" not in model_args:
        logging.info("No standard_deviance provided. Assuming default value.")
        model_args["standard_deviance"] = 1e-8

    net = PAINN_var.PainnModel_variational(
        model_args["num_interactions"],
        model_args["hidden_state_size"],
        model_args["cutoff"],
        normalize_atomwise=model_args["atomwise_normalization"],
        direct_force_output=model_args["direct_force_output"],
        standard_deviance=model_args["standard_deviance"]
    )
    net = net.to(device)

    net.load_state_dict(state_dict["model"])

    logging.info("Loading dataset.")

    if args.U0_to_E:
        logging.info("Converting U0 to E")
    dataset = data.AseDbData(
        args.dataset,
        DBTransformer.TransformRowToGraphXyzWithZpve(
            cutoff=model_args["cutoff"],
            energy_property=model_args["energy_property"],
            forces_property=model_args["forces_property"],
            U0_to_E=args.U0_to_E,
        ),
    )
    dataset = data.BufferData(dataset)  # Load data into host memory

    logging.info("Setting up test loader.")

    with open(args.split_file, "r") as fp:
        splits = json.load(fp)
        # Split the dataset
        datasplits = {}
        for key, indices in splits.items():
            datasplits[key] = torch.utils.data.Subset(dataset, indices)

    # Setup loader
    test_loader = torch.utils.data.DataLoader(
        datasplits["test"],
        1,
        collate_fn=data.CollateAtomsdata(pin_memory=device.type == "cuda"),
    )

    logging.info("Evaluating model.")

    res = torch.empty((len(datasplits["test"]),3))
    indices = torch.zeros((len(datasplits["test"]),),dtype=torch.int32)
    tot_len = len(test_loader)
    log_interval = 50

    for i, batch in enumerate(test_loader):
        if (i % log_interval == 0):
            logging.info("{}/{}".format(i,tot_len))
        device_batch = {
            k: v.to(device=device, non_blocking=True) for k, v in batch.items()
        }
        out = net.sample(device_batch, compute_stress=False, compute_forces=args.compute_forces, n_samples=args.sample_size)
        variance = model_args["standard_deviance"] ** 2 + out.var(unbiased = True)
        res[i,:] = torch.Tensor([out.mean().item(),variance.item(),batch["energy"].item()])
        indices[i] = batch["index"].item()

    res_dict = {"index": indices, "values": res}
    torch.save(res_dict,args.output_dir + "/res.pt")

    logging.info("Finished.")