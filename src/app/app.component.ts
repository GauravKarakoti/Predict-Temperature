import { Component, OnInit, Inject, PLATFORM_ID, AfterViewInit } from '@angular/core';
import { isPlatformBrowser } from '@angular/common';

import * as tf from "@tensorflow/tfjs"
import * as tfvis from "@tensorflow/tfjs-vis";

const csvurl = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ68uL8xVQ8djBJaEVIPO4wn4jBQ9ty1Lu5iTutUrvN4G_Qub__j3L0SVEp23Lu9g/pub?gid=1585054194&single=true&output=csv";

@Component({
  selector: 'app-root',
  imports: [],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss'
})
export class AppComponent implements OnInit, AfterViewInit {
  title = 'Predict-Temperature';
  dataset: any;
  isBrowser: boolean;

  constructor(@Inject(PLATFORM_ID) private platformId: Object) {
    this.isBrowser = isPlatformBrowser(platformId);
  }

  ngOnInit(): void {
    this.loadData();
  }

  ngAfterViewInit(): void {
    if (this.isBrowser) {
      this.visualizeDataset();
      this.linearregression();
    }
  }

  model= tf.sequential();

  async linearregression() {
    const numberEpochs = 100;
    const numOfFeatures= (await this.dataset.columnNames()).length - 1;
    const features:any= [];
    const target:any= [];
    const number_of_samples = 100;
    let counter = 0;

    await this.dataset.forEachAsync((e: any) => {
      if (Math.random() > 0.5 && counter < number_of_samples) {
        features.push(Object.values(e.xs));
        target.push(e.ys.Apparent_Temperature);
        counter++;
      }
    });

    const features_tensor_raw = tf.tensor2d(features, [features.length, numOfFeatures]);
    const target_tensor = tf.tensor2d(target, [target.length, 1]);

    this.model.add(tf.layers.dense({ inputShape: [1], units: 1 }));
    this.model.compile({ optimizer: 'sgd', loss: 'meanAbsoluteError' });

    const trainingplot:any= document.getElementById("training");

    // Actually training the model
    await this.model.fit(features_tensor_raw, target_tensor, {
      // batchSize: 10,
      epochs: numberEpochs,
      validationSplit: 0.2,
      callbacks: [
        tfvis.show.fitCallbacks(trainingplot, ['loss','acc', 'val_loss', 'val_acc'], {
          callbacks: ['onEpochEnd'],
        }),
        {
          onEpochEnd: async (epoch: number, logs: any) => {},
        },
        {
          onTrainEnd: async (logs: any) => {}
        }
      ]
    });
  }

  async loadData() {
    this.dataset = tf.data.csv(csvurl, {
      columnConfigs: {
        Apparent_Temperature: {
          isLabel: true
        },
      }
    });
  }

  async visualizeDataset() {
    if (!this.dataset) {
      console.error("Dataset is not loaded yet.");
      return;
    }

    const datasetplot:any = document.getElementById("datasetplotting");
    const dataset:any= [];
    const number_of_samples = 100;
    let counter = 0;
    
    await this.dataset.forEachAsync((e: any) => {
      const features = { x: e.xs.Temperature, y: e.ys.Apparent_Temperature };
      if (Math.random() > 0.5 && counter < number_of_samples) {
        dataset.push(features);
        counter++;
      }
    });

    tfvis.render.scatterplot(
      datasetplot,
      { values: dataset, series: ['Full Dataset'] },
      {
        xLabel: 'Temperature (C)',
        yLabel: 'Apparent Temperature (C)',
        // height: 300,
        // width: 420
      }
    );
  }
}