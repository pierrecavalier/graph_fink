import matplotlib.pyplot as plt
import gzip
import io
from astropy.io import fits


def plot_science(df, i):
    """Plot the science image (the image of the alert itself) of the alert n°i

    Parameters
    ----------
    df : pandas dataframe
        The dataframe containing the alerts

    i : int
        The index of the alert in the dataframe
    """
    # open the gzip file
    with gzip.open(io.BytesIO(
            df.iloc[i]['cutoutScience']['stampData']), 'rb') as f:
        # read the file
        data = f.read()
        # open the fits file
        with fits.open(io.BytesIO(data)) as hdul:
            # plot the image
            plt.imshow(hdul[0].data, cmap='gray')

            # plot the colorbar
            plt.colorbar()

            plt.title('Science of the alert n°' + str(i))


def plot_template(df, i):
    """Plot the science image (the image of the alert itself) of the alert n°i

    Parameters
    ----------
    df : pandas dataframe
        The dataframe containing the alerts

    i : int
        The index of the alert in the dataframe
    """
    # open the gzip file
    with gzip.open(io.BytesIO(
            df.iloc[i]['cutoutTemplate']['stampData']), 'rb') as f:
        # read the file
        data = f.read()
        # open the fits file
        with fits.open(io.BytesIO(data)) as hdul:
            # plot the image
            plt.imshow(hdul[0].data, cmap='gray')

            # plot the colorbar
            plt.colorbar()

            plt.title('Template of the alert n°' + str(i))


def plot_difference(df, i):
    """Plot the science image (the image of the alert itself) of the alert n°i

    Parameters
    ----------
    df : pandas dataframe
        The dataframe containing the alerts

    i : int
        The index of the alert in the dataframe
    """
    # open the gzip file
    with gzip.open(io.BytesIO(
            df.iloc[i]['cutoutDifference']['stampData']), 'rb') as f:
        # read the file
        data = f.read()
        # open the fits file
        with fits.open(io.BytesIO(data)) as hdul:
            # plot the image
            plt.imshow(hdul[0].data, cmap='gray')

            # plot the colorbar
            plt.colorbar()

            # plot a red point at location of the difference
            plt.title('Difference of the alert n°' + str(i))


def plot_lc(feature, df):
    """For each item in df plot the feature from lc_features_g
    and lc_features_r with color green and red
    and plot the average said feature for each color with a dashed line

    Parameters
    ----------
    df : pandas dataframe
        The dataframe containing the alerts

    feature : str
        The feature to plot
    """
    count_r = 0
    count_g = 0
    sum_r = 0
    sum_g = 0
    for i in range(len(df)):
        if df['lc_features_g'][i][feature] is not None:
            plt.plot(i, df['lc_features_g'][i][feature], 'g.')
            sum_g += df['lc_features_g'][i][feature]
            count_g += 1

        if df['lc_features_r'][i][feature] is not None:
            plt.plot(i, df['lc_features_r'][i][feature], 'r.')
            sum_r += df['lc_features_r'][i][feature]

            count_r += 1

    # plot a line with the average amplitude for the green
    # and red light curve with a dashed line
    plt.plot([0, len(df)], [sum_g/count_g, sum_g/count_g], '--', color='green')
    plt.plot([0, len(df)], [sum_r/count_r, sum_r/count_r], '--', color='red')
    plt.xlabel('Alerts')
    plt.ylabel(feature)
    plt.title(str(count_r) + ' red alerts and ' +
              str(count_g) + ' green alerts')
    plt.show()


def save_error(error, folder):
    plt.plot(error)
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Error in function of the number of epochs')
    plt.savefig(folder + '/error.png')
