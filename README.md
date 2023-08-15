# OzFish.AI: An AI-Powered Mobile Application to Combat Illegal Fishing in NSW

PROJECT OVERVIEW

Illegal fishing has been a serious problem in Australia. Some unintentional activities, such as taking fish illegitimately due to the inability to identify the species, might adversely affect the ecosystem. It is crucial for recreational anglers to follow fish regulations. However, it is inconvenient for them to identify fish species by referring to the brochures and measuring the fish length using rulers, which significantly lowers the overall enjoyment of recreational fishing. Besides, it is confusing for some recreational anglers or travellers to distinguish fish species with their limited fishing knowledge.  Taking the wrong fish species or exceeding the size limit may result in fines. There are some existing fishing mobile applications on the market, and very limited applications can be used to detect the fish species automatically, or they cannot provide a high accuracy rate for their detection. It can be seen that the existing applications cannot help recreational anglers solve their issues. An Android application OzFish.AI is developed to address the issues mentioned. It is easy for a recreational angler to take a picture of a fish and obtain information about the species and the size by clicking a button to determine whether it is legal to take the catch. This project will result in an extensive dataset of fish species native to New South Wales for future research, a transfer-learning-based fish classification algorithm with high accuracy, and an Android application, which will assist in storing and transforming the necessary data so that end-users can receive detailed information on fish species, size, and fish guide.

APPLICATION FUNCTIONALITIES

This Android Application contains 3 major functionalities:
1. Fish Classification (Fish.Identify)
	- This function will identify 68 NSW fish species using the device's rear camera to take a picture of the fish.
	- This picture is then classified through the application's inbuilt ML algorithm, returning to the user the identified fish species.
2. Fish Measurement (Fish.Measure)
	- This function enables the user to accurately measure their fish's length using AR technologies.
	- This function can be used in collaboration with the Fish Classification function to provide additional context to the user, such as if the measured fish meets minimimum fish length requirements for that fish species.
3. Fish Searching (Fish.Search)
	- This function enables the user to search the contained fish species database for relevant information about a particular fish species.
	- The fish database currently contains 74 NSW fish species (some are subspecies of the 68 species identified in the Fish.Identify function).

DOCUMENTATION

This repository houses a collection of three documents:
1. The coding documentation of the fish classification algorithm (README file included);
2. The coding documentation of the Android application (README file included);
3. Several demonstration videos are provided to instruct users on how to use the OzFish.AI App.
