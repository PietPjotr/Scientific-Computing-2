# Diffusion limited aggregation

This project contains 3 different implementations for the diffusion limited
aggregation

## Installation (can be skipped because we only have very widely used packages)

The requirements.txt contains all the requirements for this src folder. To
install the requirements, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

to run the main code that will present all the figures used in the report,
run the following command:

```bash
python src/main.py
```

Ror any specific file, the header comment of the file will contain the
instructions on how to run the file.

## Structure

Contains the source files for the project

src/
    main.py
    ...
    ...


Contains the figures used in the report. The subfolders are named after the
class of the figure and contain the relevant figures for that class. Some of
the function calls in main.py will save the figures in the relevant subfolder.
Thus for checking validity of the code, look for these figures. The function
calls in main.py will state where the figures will be saved.

figures/
    eta_figures/
        ...
    GrayScott/
        ...
    MC/
        ...


Contains the data files used in the report. The subfolders are named after the
class of the data and contain the relevant data files for that class.

data/
    MC/
        Timestamp/
            datafile.csv
            datafile.json

    contains the data files used in the report


Contains the necessary files for the eta analysis, and also contains a summary
file for all the different eta runs.

eta/
    clusters/
        clusterfile1.pkl
        ...
    data/
        datafile1.csv
        ...

    summary_results.csv


## License

[MIT](https://choosealicense.com/licenses/mit/)