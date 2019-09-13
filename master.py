#!/usr/bin/env python
# coding: utf-8

# # Initialization

# In[1]:


import boto3
import numpy as np
import subprocess as sp


# In[2]:


client = boto3.client('ec2')
ec2 = boto3.resource('ec2')

s3 = boto3.resource('s3')


# In[3]:


time_range = [(2018, 1), (2019, 2)]


# # Get AWS started

# ## Connect to S3

# In[4]:


# Print out bucket names
for bucket in s3.buckets.all():
    print(bucket.name)


# ## EC2 detection

# ### Find all the EC2 of interest based on their tags

# In[5]:


instances = []
tag_of_interest = 'sec_downloader'
for i in ec2.instances.all():
    print("[INFO] Found a {} {} instance".format(i.state['Name'], i.instance_type))
    if i.tags:
        for tag in i.tags:
            if tag['Key'] == 'Name' and tag['Value'] == tag_of_interest:
                if i.state['Name'] != 'terminated':
                    print("[INFO] This is actually a {} instance!\n".format(tag_of_interest))
                    instances.append(i)
                break
        
print("[INFO] Found {} instances".format(len(instances)))


# In[6]:


if len(instances) == 0:
    raise ValueError('[ERROR] No instances with that tag were found!')


# ### Instance control: turn them on if needed

# In[7]:


# Start the instances that were stopped
def start_all_instances(client, instances):
    nb_instances_started = 0
    for i in instances:
        if i.state['Name'] == 'stopped':
            response = client.start_instances(InstanceIds=[i.id])
            nb_instances_started += 1
        elif i.state['Name'] != 'running':
            print("[ERROR] Instance {} is neither stopped nor running. Cannot start it.".format(i.id))
    return nb_instances_started
            
# Stop the instances that were started
def stop_all_instances(client, instances):
    nb_instances_stopped = 0
    for i in instances:
        if i.state['Name'] == 'running':
            response = client.stop_instances(InstanceIds=[i.id])
            nb_instances_stopped += 1
        elif i.state['Name'] != 'stopped':
            print("[ERROR] Instance {} is neither stopped nor running. Cannot start it.".format(i.id))
    return nb_instances_stopped


# In[8]:


nb_instances_started = start_all_instances(client, instances)
print("[INFO] Number of instances restarted: {} | Total running: {}"
      .format(nb_instances_started, len(instances)))


# # Setup the work to be done

# ## Split the work among all the instances that we have turned on

# In[9]:


def create_qtr_list(time_range):
    # Sanity checks
    assert len(time_range) == 2
    assert 1994 <= time_range[0][0] and 1994 <= time_range[1][0]
    assert 1 <= time_range[0][1] <= 4 and 1 <= time_range[1][1] <= 4
    assert time_range[1][0] >= time_range[0][0]
    if time_range[1][0] == time_range[0][0]:  # Same year
        assert time_range[1][1] >= time_range[0][1]  # Need different QTR
    
    list_qtr = []
    for year in range(time_range[0][0], time_range[1][0]+1):
        for qtr in range(1, 5):
            # Manage the start and end within a year
            if year == time_range[0][0]:
                if qtr < time_range[0][1]:
                    continue
            if year == time_range[1][0]:
                if qtr > time_range[1][1]:
                    break
            
            # Common case
            list_qtr.append((year, qtr))
    
    # Sanity checks
    assert list_qtr[0] == time_range[0]
    assert list_qtr[-1] == time_range[1]
    return list_qtr

def test_create_qtr_list():
    test_1 = create_qtr_list([(2018, 1), (2018, 4)])
    assert test_1 == [(2018, 1), (2018, 2), (2018, 3), (2018, 4)]
    test_2 = create_qtr_list([(2016, 2), (2017, 3)])
    assert test_2 == [(2016, 2), (2016, 3), (2016, 4), (2017, 1), (2017, 2), (2017, 3)]
    return True
test_create_qtr_list()


# In[10]:


def yearly_qtr_list(time_range):
    year_list = []
    if time_range[0][0] == time_range[1][0]:
        year_list = create_qtr_list(time_range)
    else:
        for year in range(time_range[0][0], time_range[1][0]+1):
            if year == time_range[0][0]:
                year_list.append(create_qtr_list([(year, time_range[0][1]), (year, 4)]))
            elif year == time_range[1][0]:
                year_list.append(create_qtr_list([(year, 1), (year, time_range[1][1])]))
            else:
                year_list.append(create_qtr_list([(year, 1), (year, 4)]))
    return year_list

def test_yearly_qtr_list():
    test_1 = yearly_qtr_list([(2016, 2), (2016, 2)])
    assert test_1 == [(2016, 2)]
    test_2 = yearly_qtr_list([(2015, 2), (2016, 3)])
    assert test_2 == [[(2015, 2), (2015, 3), (2015, 4)], [(2016, 1), (2016, 2), (2016, 3)]]
    return True
test_yearly_qtr_list()


# In[11]:


def split_work_among_instances(time_range, instances):
    # time_range is a simple list containing the start & end tuple
    # instances is the list of instances object from AWS
    # returns a simple list of (time_range, IP) to distribute the work

    # Create the list of quarters
    all_qtr = create_qtr_list(time_range)
    
    # Do not use more instances than work packages available
    nb_instances = min(len(all_qtr), len(instances))
    instances = instances[:nb_instances]
    #print("nb_instances:", len(instances), "| instances actually used:", nb_instances)
    
    # Split the work equally
    qtr_indexes = np.linspace(0, len(all_qtr), len(instances), endpoint=False)
    #print(qtr_indexes)
    qtr_indexes = [int(i) for i in qtr_indexes]  # Cast to int
    #print(qtr_indexes)
    qtr_indexes.append(len(all_qtr))  # Add the last element for the comprehension below
    #print(qtr_indexes)
    split_work = []
    for i in range(len(qtr_indexes)-1):
        split_work.append(all_qtr[qtr_indexes[i]:qtr_indexes[i+1]])
    #split_work = [all_qtr[qtr_indexes[i]:qtr_indexes[i+1]] for i in qtr_indexes if i<len(qtr_indexes)-1]  # ignore last index to prevent overflow
    #print(split_work)
    """
    for instance_count in range(len(instances)):
        #print(instance_count)
        print(split_work[instance_count])
        print()
    """
    
    return split_work

def test_split_work_among_instances():
    test_1 = split_work_among_instances([(2010, 1), (2019, 2)], [1, 2, 3, 4])
    assert test_1 == [[(2010, 1), (2010, 2), (2010, 3), (2010, 4), (2011, 1), (2011, 2), (2011, 3), (2011, 4), (2012, 1)], 
                      [(2012, 2), (2012, 3), (2012, 4), (2013, 1), (2013, 2), (2013, 3), (2013, 4), (2014, 1), (2014, 2), (2014, 3)], 
                      [(2014, 4), (2015, 1), (2015, 2), (2015, 3), (2015, 4), (2016, 1), (2016, 2), (2016, 3), (2016, 4)], 
                      [(2017, 1), (2017, 2), (2017, 3), (2017, 4), (2018, 1), (2018, 2), (2018, 3), (2018, 4), (2019, 1), (2019, 2)]]
    
    # Test for 1 instance ---> more instances than work packages
    time_range = [(2000, 1), (2002, 2)]
    all_qtr = create_qtr_list(time_range)
    for nb_instances in range(1, len(all_qtr)+5):  # verify that too many instances are handled properly
        test = split_work_among_instances(time_range, [1]*nb_instances)
        #print(test)
        #print("Instances used:", min(len(all_qtr), nb_instances))
        assert len(test) == min(len(create_qtr_list(time_range)), nb_instances)
        
    return True

test_split_work_among_instances()  


# In[12]:


list_of_work = split_work_among_instances(time_range, instances)
list_of_work


# # SSH to the EC2 instances and prepare them

# In[13]:


instances[0].public_ip_address


# In[15]:


bash_script = '/home/alex/Desktop/Insight project/launch_instance.sh'
for instance_nb, i in enumerate(instances):
    print("[INFO] SSH to {} | Sending the following task:\n{}".format(i.public_ip_address, list_of_work[instance_nb]))
    if i.public_ip_address != None:
        sp.check_call([bash_script, '-ip', str(i.public_ip_address), '-tr', str(list_of_work[instance_nb])])
    else:
        raise ValueError('[ERROR] IP address cannot be None!')


# In[ ]:




