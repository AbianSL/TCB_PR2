from src.csv_extractor import CsvExtractor
import src.manage_data as manage_data
import src.analyse_data as analyse_data
import src.figures as figures
import src.model_data as model_data


def main():

    # Leer las velocidades radiales
    rjd_rv, rv, srv, instrumento_rv = manage_data.opendata("data/Epsilon_Eridani_RVdata.csv", "RV")

    # Leer el índice de actividad
    rjd_s, s, ss, instrumento_s = manage_data.opendata("data/Epsilon_Eridani_SMW.csv", "SMW")

    # Comparar una constante y una tendencia lineal
    model_data.fit_rv_trends(rjd_rv, rv, srv, instrumento_rv)
    model_data.fit_rv_sine(rjd_rv, rv, srv, instrumento_rv)

    # Copia de la actividad filtrada por instrumento
    rjd_s_clean, s_clean, ss_clean, instrumento_s_clean = (manage_data.clean_data(rjd_s, s, ss, instrumento_s))

    analyse_data.compare_activity_periodograms(rjd_s, s, ss, instrumento_s, rjd_s_clean, s_clean, ss_clean, instrumento_s_clean)
    analyse_data.analyse_rv_periodograms(rjd_rv, rv, srv, instrumento_rv)
    # Analizar el muestreo de cada instrumento
    analyse_data.analyse_window(rjd_s, instrumento_s)

    # Describir ambas series
    analyse_data.describe_data(rjd_rv, rv, srv, instrumento_rv, "RV (m/s)")
    analyse_data.describe_data(rjd_s, s, ss, instrumento_s, "S_MW (adimensional)")

    #Para ver cuantos puntos eliminamos por serie (5 sigmas)
    analyse_data.inspect_outliers(rjd_rv, rv, instrumento_rv, "RV")
    analyse_data.inspect_outliers(rjd_s, s, instrumento_s, "S_MW")

    analyse_data.inspect_region(rjd_s, s, ss, instrumento_s,"HARPS03_1", 3304.69, 3304.74)
    analyse_data.inspect_region(rjd_s, s, ss, instrumento_s,"ESPRESSO18_1", 8488, 8491)

    # Representar la actividad
    figures.plot_activity(rjd_s, s, ss, instrumento_s)
    figures.plot_rv(rjd_rv, rv, srv, instrumento_rv)


if __name__ == "__main__":
    main()