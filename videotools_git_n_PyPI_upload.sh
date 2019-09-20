#!/bin/bash

# Automatic selected upload of the folder content to GitHub & PyPI
# Default is to make a minor update but this can be changed using the -l argument.
# As always, GitHub commit message is mandatory.

version_level=2  # By default, minor revisions
commitText='No message'
package='videotools'  # Default package so far
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    -m|--message)
    commitText="$2"
    shift # past argument
    shift # past value
    ;;
    -l|--level)
    version_level="$2"
    shift # past argument
    shift # past value
    ;;
    -d|--folder)
    package="$2"
    shift # past argument
    shift # past value
    ;;
esac
done

if [ "$commitText" = 'No message' ] # WARNING: all spaces/blanks very important here
then
    echo '[ERROR] Commit message required. No action taken.'
elif [ "$commitText" = '' ]
then
    echo '[ERROR] Commit message is empty. No action taken.'
else
    # Create PyPi package
    new_version=$(python3 increment_setup_version.py --level $version_level --folder $package) # Better than backticks
    cd $package
    python3 setup.py sdist bdist_wheel
    # Upload to PyPi latest version
    twine upload dist/$package-$new_version*
    
    # Upload to GitHub, including latest package
    rm -rf __pycache__/
    git add -u :/
    git commit -m "$commitText"
    git push git@github.com:AlexBdx/$package.git master
fi



