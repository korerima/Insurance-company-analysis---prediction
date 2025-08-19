from data_preprocessing import prepocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Data preprocessing script")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Path to save processed CSV file")
    args = parser.parse_args()

    preprocessor = prepocess()
    df = preprocessor.load_data(args.input)
    df_processed = preprocessor.handle_dataset(df, args.output)
    print(f"Processed dataset saved to {args.output}")

if __name__ == "__main__":
    main()
