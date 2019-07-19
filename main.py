# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from flask import Flask, render_template, url_for, request, redirect,flash

import logging
import random
import numpy as np
import cv2
import mxnet as mx
from ocr.utils.iam_dataset import resize_image
from ocr.paragraph_segmentation_dcnn import SegmentationNetwork, paragraph_segmentation_transform
from ocr.word_and_line_segmentation import SSD as WordSegmentationNet, predict_bounding_boxes
from ocr.handwriting_line_recognition import Network as HandwritingRecognitionNet, handwriting_recognition_transform
from ocr.handwriting_line_recognition import decode as decoder_handwriting, alphabet_encoding

from ocr.utils.iam_dataset import crop_handwriting_page
from ocr.utils.word_to_line import sort_bbs_line_by_line, crop_line_images

from google.cloud import language
from google.cloud import translate

import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ""

# Instantiates a client
client = language.LanguageServiceClient()
translate_client = translate.Client()

app = Flask(__name__)
app.secret_key = -

# Setup
logging.basicConfig(level=logging.DEBUG)
random.seed(123)
np.random.seed(123)
mx.random.seed(123)

# Input sizes
segmented_paragraph_size = (800, 800)
line_image_size = (60, 800)

# Parameters
min_c = 0.01
overlap_thres = 0.001
topk = 400
rnn_hidden_states = 512
rnn_layers = 2
max_seq_len = 160

recognition_model = -
paragraph_segmentation_model = -
word_segmentation_model = -


@app.route("/")
def home():
    return render_template("home.html")


@app.route('/about', methods=['GET', 'POST'])
def about():
    if request.method == 'POST':
        # do stuff when the form is submitted

        # redirect to end the POST handling
        # the redirect can be to the same route or somewhere else
        return redirect(url_for('about.html'))
    return render_template('about.html')


# Google Cloud Sentiment Analysis
def gc_sentiment(text):
    document = language.types.Document(
        content=text,
        type=language.enums.Document.Type.PLAIN_TEXT)
    annotations = client.analyze_sentiment(document=document)
    score = annotations.document_sentiment.score
    magnitude = annotations.document_sentiment.magnitude
    return score, magnitude


# Google Cloud translate function
def gc_translate(text):
    target = 'zh'
    document = language.types.Document(
        content=text,
        type=language.enums.Document.Type.PLAIN_TEXT)
    result = translate_client.translate(
        text, target_language=target)
    return result


def get_arg_max(prob):
    '''
    The greedy algorithm convert the output of the handwriting recognition network
    into strings.
    '''
    arg_max = prob.topk(axis=2).asnumpy()
    return decoder_handwriting(arg_max)[0]


denoise_func = get_arg_max


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # read image file string data
        if 'file' not in request.files:
            return redirect(url_for('home'))
        else:
            filestr = request.files['file'].read()
            # convert string data to numpy array
            npimg = np.fromstring(filestr, np.uint8)
            # convert numpy array to image
            img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
            ctx = ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu()

            # Models
            paragraph_segmentation_net = SegmentationNetwork(ctx=ctx)
            paragraph_segmentation_net.cnn.load_parameters(paragraph_segmentation_model, ctx)

            word_segmentation_net = WordSegmentationNet(2, ctx=ctx)
            word_segmentation_net.load_parameters(word_segmentation_model, ctx)

            handwriting_line_recognition_net = HandwritingRecognitionNet(rnn_hidden_states=rnn_hidden_states,
                                                                         rnn_layers=rnn_layers,
                                                                         max_seq_len=max_seq_len,
                                                                         ctx=ctx)
            handwriting_line_recognition_net.load_parameters(recognition_model, ctx)

            MAX_IMAGE_SIZE_FORM = (1120, 800)

            img_arr = np.asarray(img)

            resized_image = paragraph_segmentation_transform(img_arr, image_size=MAX_IMAGE_SIZE_FORM)
            paragraph_bb = paragraph_segmentation_net(resized_image.as_in_context(ctx))
            paragraph_segmented_image = crop_handwriting_page(img_arr, paragraph_bb[0].asnumpy(),
                                                              image_size=segmented_paragraph_size)
            word_bb = predict_bounding_boxes(word_segmentation_net, paragraph_segmented_image, min_c, overlap_thres, topk,
                                             ctx)
            line_bbs = sort_bbs_line_by_line(word_bb)
            line_images = crop_line_images(paragraph_segmented_image, line_bbs)

            predicted_text = []

            for line_image in line_images:
                line_image = handwriting_recognition_transform(line_image, line_image_size)
                character_probabilities = handwriting_line_recognition_net(line_image.as_in_context(ctx))
                decoded_text = denoise_func(character_probabilities)
                predicted_text.append(decoded_text)
            text = ' '.join(predicted_text)
            print(text)
            translated = gc_translate(text)
            sentiment = gc_sentiment(text)
            print(translated)
            print(sentiment)
            text_dict = {"text": text,
                         "translated": translated['translatedText'],
                         "polarity": sentiment[0],
                         "magnitude": sentiment[1]}

            a = text_dict
            return render_template('result.html', prediction=a)


if __name__ == "__main__":
    app.run(debug=True)
