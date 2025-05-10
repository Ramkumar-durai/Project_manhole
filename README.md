# Project_manhole
Using YOLO Technology for real time manhole detection, especially made for TamilNadu Municipal Corporation.
from flask import Flask, render_template, redirect, url_for, session, request, flash, Response, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import cv2
from ultralytics import YOLO
import threading
import time
