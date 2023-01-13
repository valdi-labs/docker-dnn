from uplink_python.uplink import Uplink
from uplink_python.errors import StorjException
from uplink_python.module_classes import ListObjectsOptions

if __name__ == "__main__":
    # Storj configuration information
    MY_API_KEY = "my-api-key"
    MY_SATELLITE = "my-satellite-url"
    MY_BUCKET = "my-bucket-name"
    MY_ENCRYPTION_PASSPHRASE = "my-custom-passphrase"

    try:
        # create an object of Uplink class
        uplink = Uplink()

        # function calls
        # request access using passphrase
        print("\nRequesting Access using passphrase...")
        access = uplink.request_access_with_passphrase(MY_SATELLITE, MY_API_KEY,
                                                       MY_ENCRYPTION_PASSPHRASE)
        print("Request Access: SUCCESS!")

        # open Storj project
        print("\nOpening the Storj project, corresponding to the parsed Access...")
        project = access.open_project()
        print("Desired Storj project: OPENED!")
        #

        # list objects in given bucket with above options or None
        print("\nListing object's names...")
        objects_list = project.list_objects(MY_BUCKET, ListObjectsOptions(recursive=True,
                                                                          system=True))
        # print all objects path
        for obj in objects_list:
            print(obj.key, " | ", obj.is_prefix)  # as python class object
            print(obj.get_dict())  # as python dictionary
        print("Objects listing: COMPLETE!")
    except StorjException as exception:
        print("Exception Caught: ", exception.details)
