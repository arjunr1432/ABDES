################################################################
####### Automate Block Diagram Evaluation System (ABDES) #######
############# Training related logics defined here.#############
################################################################

# Imports
from CommonMethods import processImage, generateInitialClusterPoints, generateAllContourPointsForClustering, \
    image_file_dir_path, data_save_dir_path
from ConsoleOutMethods import displayClusters
from CustomClasses import TrainedData
from Kmeans import kmeans
import matplotlib.pyplot as plt
# from dill import dill
import dill

def trainData(images_names_to_train):
    images = []
    for image_name in images_names_to_train:
        images.append(processImage(image_file_dir_path, image_name))
    # images.append(processImage('images/diag2.bmp'))

    fig = plt.figure(figsize=(8, 6))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
    colors = ['green', 'blue', 'pink', 'gold', 'red', 'yellow', 'black']

    # Populate initial points for clustering
    init_contour_points = generateInitialClusterPoints(images)

    # Generate some points
    points = generateAllContourPointsForClustering(images)
    # When do we say the optimization has 'converged' and stop updating clusters
    opt_cutoff = 0.5

    # Cluster those data!
    clusters = kmeans(init_contour_points, points, opt_cutoff)
    displayClusters(clusters)

    saveTrainingData(images, clusters)

    # Draw the cluster points
    plt.axis([0, 1024, 0, 768])
    for i, c in enumerate(clusters):
        X = []
        Y = []
        Xc = [c.centroid.coords[0]]
        Yc = [c.centroid.coords[1]]
        for p in c.points:
            X.append(p.coords[0])
            Y.append(p.coords[1])
        plt.plot(X, Y, 'w', markerfacecolor=colors[i], marker='.', markersize=10)
        plt.plot(Xc, Yc, 'o', markerfacecolor=colors[i], marker='*', markeredgecolor='k', markersize=10)
    plt.show()


# The main method from where the all project execution begins.
# arguments : void
# return : void
def saveTrainingData(images, clusters):
    trained_data = TrainedData(images, clusters)

    with open(data_save_dir_path + 'trained_data.pkl', 'wb') as f:
        dill.dump(trained_data, f)
    pass
