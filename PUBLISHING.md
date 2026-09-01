## Publishing to PyPI

```bash
pip install build twine

cd /tmp
python -m build /local/malva/repos/malva --outdir /local/malva/repos/malva/dist
python -m twine upload /local/malva/repos/malva/dist/*
```