 # Create directory to save embeddings
    import os

    embeddings_dir = Path("embeddings_cache")
    os.makedirs(embeddings_dir, exist_ok=True)
    print(f"Saving embeddings to: {embeddings_dir}")

    with torch.no_grad():
        for i, subj_id in enumerate(participants_ids):
            print(f"Processing subject {i+1}/{len(participants_ids)}: {subj_id}...")

            # Load this subject's data (uses npz cache if available)
            data = fast_load_meg_data(
                subj_id, cache_dir="meg_cache", base_path=REST_PATH
            )

            # Generate embeddings in batches
            individual_embeddings = []
            infer_dataset = MEG_Dataset(
                data,
                window_size=2000,
                transforms=transforms.Compose([ZScore()]),
                stride=500,
            )
            infer_dataloader = DataLoader(
                infer_dataset, batch_size=256, shuffle=False, num_workers=2
            )

            for batch_idx, (x1, x2) in enumerate(infer_dataloader):
                x1 = x1.to(device)
                z1 = encoder(x1)
                individual_embeddings.append(z1.cpu())

            # Concatenate and save to disk immediately
            subject_embeddings = torch.cat(individual_embeddings, dim=0)
            torch.save(subject_embeddings, embeddings_dir / f"{subj_id}_embeddings.pt")
            print(
                f"  Generated {subject_embeddings.shape[0]} embeddings, saved to disk"
            )

            # Free memory
            del data, individual_embeddings, subject_embeddings
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"\n✓ Generated and saved embeddings for {len(participants_ids)} subjects")
    print("=" * 50)

    # |export

    print("\n" + "=" * 50)
    print("LOADING METADATA FOR VISUALIZATION")
    print("=" * 50)

    # Load age and sex from participants.tsv
    age_labels_list = []
    sex_labels_list = []

    for i, subj_id in enumerate(participants_ids):
        # Get metadata from dataframe
        subj_data = df[df["participant_id"] == subj_id].iloc[0]
        age = subj_data.get("age", 0)  # Adjust column name as needed
        sex = subj_data.get("sex", 0)  # Adjust column name as needed (or 'gender')

        # Load embeddings to get count
        embeddings = torch.load(embeddings_dir / f"{subj_id}_embeddings.pt")
        num_windows = embeddings.shape[0]

        age_labels_list.append(np.full(num_windows, age))
        sex_labels_list.append(np.full(num_windows, sex))

        del embeddings  # Free memory
        if (i + 1) % 50 == 0:
            print(f"Processed metadata for {i+1}/{len(participants_ids)} subjects")

    age_labels_numpy = np.concatenate(age_labels_list, axis=0)
    sex_labels_numpy = np.concatenate(sex_labels_list, axis=0)
    print(f"Total data points: {len(age_labels_numpy)}")

    print("\n" + "=" * 50)
    print("DIMENSIONALITY REDUCTION (UMAP)")
    print("=" * 50)

    print("Fitting UMAP reducer in batches...")
    dim_reducer = umap.UMAP(
        n_neighbors=30, min_dist=0.1, n_components=2, random_state=42
    )

    # Load embeddings in batches to avoid RAM overflow
    batch_size_subjects = 50  # Process 50 subjects at a time
    all_embeddings_2d = []

    for batch_start in range(0, len(participants_ids), batch_size_subjects):
        batch_end = min(batch_start + batch_size_subjects, len(participants_ids))
        print(
            f"Processing subjects {batch_start+1}-{batch_end}/{len(participants_ids)}..."
        )

        batch_embeddings = []
        for i in range(batch_start, batch_end):
            subj_id = participants_ids[i]
            embeddings = torch.load(embeddings_dir / f"{subj_id}_embeddings.pt")
            batch_embeddings.append(embeddings)

        batch_concat = torch.cat(batch_embeddings).numpy()

        # Fit on first batch, transform on subsequent
        if batch_start == 0:
            batch_2d = dim_reducer.fit_transform(batch_concat)
        else:
            batch_2d = dim_reducer.transform(batch_concat)

        all_embeddings_2d.append(batch_2d)
        del batch_embeddings, batch_concat
        print(f"  Batch shape: {batch_2d.shape}")

    embedding_2d = np.vstack(all_embeddings_2d)
    print(f"\nFinal UMAP output shape: {embedding_2d.shape}")
    print("✓ UMAP reduction complete")
    print("=" * 50)

    # |export

    print("\n" + "=" * 50)
    print("CREATING VISUALIZATIONS")
    print("=" * 50)

    # --- Create a 1x2 figure ---
    print("Generating plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # --- Plot 1: Colored by AGE ---
    plot1 = ax1.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=age_labels_numpy,  # Color by age
        cmap="jet",  # Continuous colormap
        s=0.5,
        alpha=0.3,
    )
    ax1.set_title("UMAP Projection Colored by Age")
    ax1.set_xlabel("UMAP Dimension 1")
    ax1.set_ylabel("UMAP Dimension 2")
    fig.colorbar(plot1, ax=ax1, label="Participant Age")

    # --- Plot 2: Colored by SEX ---
    # (Assuming sex=0 and sex=1)
    plot2 = ax2.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=sex_labels_numpy,  # Color by sex
        cmap="coolwarm",  # Categorical colormap
        s=0.5,
        alpha=0.3,
    )
    ax2.set_title("UMAP Projection Colored by Sex")
    ax2.set_xlabel("UMAP Dimension 1")
    ax2.set_ylabel("UMAP Dimension 2")
    fig.colorbar(plot2, ax=ax2, label="Participant Sex")

    print("✓ Plots generated")
    plt.show()

    print("\n" + "=" * 50)
    print("ALL PROCESSING COMPLETE")
    print("=" * 50)

    # Optional: Visualization code (commented out as it references undefined variables)
    # transform = transforms.Compose([ZScore(), StdGaussianNoise(std=0.5)])
    # norm_transform = transforms.Compose([ZScore()])
    # plt.plot(norm_transform(data[:1000,0]))
    # plt.plot(transform(data[:1000,0]))
    # plt.show()

    # from nbdev.export import nb_export
    # nb_export('understand.ipynb', '.')