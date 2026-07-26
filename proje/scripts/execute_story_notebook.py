"""Projenin veri hikâyeleştirme notebook'unu baştan sona çalıştırır."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient

from scripts.project_config import PROJECT_ROOT


NOTEBOOK_PATH = (
    PROJECT_ROOT / "notebooks" / "retail_demand_forecasting_story.ipynb"
)


def main() -> None:
    if not NOTEBOOK_PATH.is_file():
        raise FileNotFoundError(f"Notebook bulunamadı: {NOTEBOOK_PATH}")

    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}},
    )
    executed = client.execute()
    nbformat.write(executed, NOTEBOOK_PATH)
    code_cells = [
        cell for cell in executed.cells if cell.get("cell_type") == "code"
    ]
    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    if errors:
        raise RuntimeError(f"Notebook {len(errors)} hata çıktısı içeriyor.")
    print(
        f"Notebook başarıyla çalıştırıldı: {NOTEBOOK_PATH} "
        f"({len(code_cells)} kod hücresi, 0 hata)"
    )


if __name__ == "__main__":
    main()
