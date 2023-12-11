import argparse

from leafs_clippers.ayann.tracer import Tracer


def main(
    tracer_path: str, species_path: str = "data/species.txt", no_csv: bool = False
) -> None:
    tracer = Tracer(tracer_path=tracer_path, species_path=species_path)

    abundances = tracer.abundances
    abundances.sort_values("Xnuc", ascending=False, inplace=True)
    print(abundances)

    if not no_csv:
        abundances.to_csv(f"{tracer_path}.csv", index=False)


def cli() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "tracer_path",
        type=str,
        help="Path to tracer file.",
    )
    parser.add_argument(
        "--species_path",
        type=str,
        help="Path to species file.",
        default="data/species.txt",
    )
    parser.add_argument(
        "--no_csv",
        action="store_true",
        help="Do not save abundances to csv.",
    )

    args = parser.parse_args()
    main(
        tracer_path=args.tracer_path,
        species_path=args.species_path,
        no_csv=args.no_csv,
    )


if __name__ == "__main__":
    cli()
