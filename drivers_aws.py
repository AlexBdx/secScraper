import boto3
from termcolor import colored

ec2 = boto3.resource('ec2')
#print("[INFO] There are {} instances listed".format(len(ec2.instances.all())))

for i in ec2.instances.all():

    print("Id: {0}\tState: {1}\tLaunched: {2}\tRoot Device Name: {3}".format(
        colored(i.id, 'cyan'),
        colored(i.state['Name'], 'green'),
        colored(i.launch_time, 'cyan'),
        colored(i.root_device_name, 'cyan')
    ))

    print("\tArch: {0}\tHypervisor: {1}".format(
        colored(i.architecture, 'cyan'),
        colored(i.hypervisor, 'cyan')
    ))

    print("\tPriv. IP: {0}\tPub. IP: {1}".format(
        colored(i.private_ip_address, 'red'),
        colored(i.public_ip_address, 'green')
    ))

    print("\tPriv. DNS: {0}\tPub. DNS: {1}".format(
        colored(i.private_dns_name, 'red'),
        colored(i.public_dns_name, 'green')
    ))

    print("\tSubnet: {0}\tSubnet Id: {1}".format(
        colored(i.subnet, 'cyan'),
        colored(i.subnet_id, 'cyan')
    ))

    print("\tKernel: {0}\tInstance Type: {1}".format(
        colored(i.kernel_id, 'cyan'),
        colored(i.instance_type, 'cyan')
    ))

    print("\tRAM Disk Id: {0}\tAMI Id: {1}\tPlatform: {2}\t EBS Optimized: {3}".format(
        colored(i.ramdisk_id, 'cyan'),
        colored(i.image_id, 'cyan'),
        colored(i.platform, 'cyan'),
        colored(i.ebs_optimized, 'cyan')
    ))

    print("\tBlock Device Mappings:")
    for idx, dev in enumerate(i.block_device_mappings, start=1):
        print("\t- [{0}] Device Name: {1}\tVol Id: {2}\tStatus: {3}\tDeleteOnTermination: {4}\tAttachTime: {5}".format(
            idx,
            colored(dev['DeviceName'], 'cyan'),
            colored(dev['Ebs']['VolumeId'], 'cyan'),
            colored(dev['Ebs']['Status'], 'green'),
            colored(dev['Ebs']['DeleteOnTermination'], 'cyan'),
            colored(dev['Ebs']['AttachTime'], 'cyan')
        ))

    if i.tags:
        print("\tTags:")
        for idx, tag in enumerate(i.tags, start=1):
            print("\t- [{0}] Key: {1}\tValue: {2}".format(
                idx,
                colored(tag['Key'], 'cyan'),
                colored(tag['Value'], 'cyan')
            ))

    print("\tProduct codes:")
    for idx, details in enumerate(i.product_codes, start=1):
        print("\t- [{0}] Id: {1}\tType: {2}".format(
            idx,
            colored(details['ProductCodeId'], 'cyan'),
            colored(details['ProductCodeType'], 'cyan')
        ))

    print("Console Output:")
    # Commented out because this creates a lot of clutter..
    # print(i.console_output()['Output'])

    print("--------------------")
