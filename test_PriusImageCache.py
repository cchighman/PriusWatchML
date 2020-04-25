import pytest
from datetime import datetime, timedelta
from abc import ABC, ABCMeta, abstractmethod
from PriusImageCache import Deduplication, PathDeduplication, ImageDeduplication

path1 = "./time_id.jpg"
newPath1 = "./laterTime_id.jpg"

path2 = "./car1.jpg"
image_hash = '068309071b5b49490096090f2530ffff'
expected_hash = '068309071b5b49490096090f2530ffff'


@pytest.fixture(scope="module")
def path_dedup():
	return PathDeduplication()

def image_dedup():
	return ImageDeduplication()

def test_instance(path_dedup):
	assert (isinstance(path_dedup, PathDeduplication))

def test_get_image_hash(path_dedup):
	result_hash = path_dedup.get_image_hash(path1)
	return(image_hash == result_hash)

def test_get_hash(path_dedup):
	path_dedup.put_hash(path1)
	result_hash = path_dedup.get_hash(path1)
	assert result_hash == expected_hash

def test_put_hash(path_dedup):
	path_dedup.put_hash(path1)
	result_hash = path_dedup.get_hash(path1)
	assert result_hash == expected_hash


def test_compare_images_when_duplicate(path_dedup):
	result = path_dedup.compare_images(path1, path1)
	assert result


def test_compare_images_when_not_duplicate(path_dedup):
	result = path_dedup.compare_images(path1, path2)
	assert result is False


def test_is_image_duplicate_when_duplicate(path_dedup):
	path_dedup.put_hash(path1)
	result = path_dedup.is_image_duplicate(path1)
	assert result


def test_is_image_duplicate_when_not_duplicate(path_dedup):
	path_dedup.put_hash(path1)
	result = path_dedup.is_image_duplicate(newPath1)
	assert result is False
