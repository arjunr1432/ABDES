################################################################
####### Automate Block Diagram Evaluation System (ABDES) #######
####### Methods for printing data in console defined here.######
################################################################


def displayClusters(clusters):
    for i, cluster in enumerate(clusters):
        print ("Cluster " + str(i + 1) + " : Text : " + cluster.points[0].text)
        for point in cluster.points:
            print ("( " + str(point.coords) + " )")


def displayArrows(arrows):
    cnt = 1
    for arrow in arrows:
        print ("Arrow", cnt, "Head : ", arrow.head.point, "Tail : ", arrow.tail.point)
        cnt += 1


def displayClassificationResult(point_cluster_map):
    for i, (point, cluster) in enumerate(point_cluster_map):
        if cluster:
            string_buffer = "Point " + str(i + 1) + ": " + str(point.coords) + "{" + point.text + "}" + " -> "
            string_buffer += "Cluster [ "
            for point in cluster.points:
                string_buffer += "( " + str(point.coords) + " )"
            string_buffer += " ] { " + cluster.points[0].text + " }"
            print (string_buffer)