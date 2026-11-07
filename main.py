from src.csv_extractor import CsvExtractor
import src.manage_data as manage_data
import src.analyse_data as analyse_data
import src.figures as figures


def main():

    # Leer las velocidades radiales
    rjd_rv, rv, srv, instrumento_rv = manage_data.opendata("data/Epsilon_Eridani_RVdata.csv", "RV")

    # Leer el índice de actividad
    rjd_s, s, ss, instrumento_s = manage_data.opendata("data/Epsilon_Eridani_SMW.csv", "SMW")

    # Describir ambas series
    analyse_data.describe_data(rjd_rv, rv, srv, instrumento_rv, "RV (m/s)")
    analyse_data.describe_data(rjd_s, s, ss, instrumento_s, "S_MW (adimensional)")

    #Para ver cuantos puntos eliminamos por serie (5 sigmas)
    analyse_data.inspect_outliers(rjd_rv, rv, instrumento_rv, "RV")
    analyse_data.inspect_outliers(rjd_s, s, instrumento_s, "S_MW")

    # Representar la actividad
    figures.plot_activity(rjd_s, s, ss, instrumento_s)
    figures.plot_rv(rjd_rv, rv, srv, instrumento_rv)


if __name__ == "__main__":
    main()