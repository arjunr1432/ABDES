################################################################
####### Automate Block Diagram Evaluation System (ABDES) #######
########## K-Means Classification logics defined below.#########
################################################################

# Imports
from CommonMethods import getDistance


def kmeansClassification(clusters, points):
    cluster_count = len(clusters)
    point_cluster_map = []
    # For every point in the dataset ...
    for p in points:
        # Get the distance between that point and the centroid of the first
        # cluster.
        smallest_distance = getDistance(p, clusters[0].centroid)

        # Set the cluster this point belongs to
        cluster_index = 0

        # For the remainder of the clusters ...
        for i in range(cluster_count - 1):
            # calculate the distance of that point to each other cluster's
            # centroid.
            distance = getDistance(p, clusters[i + 1].centroid)

            if distance < smallest_distance:
                smallest_distance = distance
                cluster_index = i + 1
        if smallest_distance > 50:
            print ("Removed")
            point_cluster_map.append((p, None))
        else:
            point_cluster_map.append((p, clusters[cluster_index]))
    return point_cluster_map
