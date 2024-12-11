from data.CNR_Outdoor.data_utils import load_pdr_dataset, load_gps_dataset

def main():
    # Load PDR dataset
    pdr_dataset = load_pdr_dataset()

    # Load GPS dataset
    gps_dataset = load_gps_dataset()

    print(pdr_dataset)
    print(gps_dataset)

if __name__ == "__main__":
    main()
