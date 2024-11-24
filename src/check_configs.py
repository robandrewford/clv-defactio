from config.config_loader import verify_configs

if __name__ == "__main__":
    try:
        configs = verify_configs()
        
        # Print some basic info about each config
        for config_name, config_data in configs.items():
            print(f"\n{config_name.upper()} Configuration:")
            print("-" * 40)
            for key in config_data.keys():
                print(f"- {key}")
                
    except Exception as e:
        print(f"Configuration check failed: {str(e)}") 