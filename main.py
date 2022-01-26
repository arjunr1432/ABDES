################################################################
####### Automate Block Diagram Evaluation System (ABDES) #######
############## Project Execution begins from here ##############
################################################################

# Imports
from ArrowDetection import arrowEvaluation
from Evaluation import diagramEvaluation
from Training import trainData

if __name__ == "__main__":
    images_names_to_train = ["diag1.jpg"]
    image_name_for_evaluation = "diag3_excess.jpggi"
    # trainData(images_names_to_train)
    diagramEvaluation(image_name_for_evaluation)
    # arrowEvaluation(image_name_for_evaluation)
