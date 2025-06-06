import { Component, Inject, PLATFORM_ID, AfterViewInit } from '@angular/core';
import { isPlatformBrowser } from '@angular/common';
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

const CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRm-d5gwpY6E-NYgp95ycNmQzPvQ8fAh5MgOI7Tn_Podim_OVBjn168oWAEQVSq2w/pub?gid=971307772&single=true&output=csv";

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements AfterViewInit {
  dataArray: any[] = [];
  isBrowser: boolean;
  model = tf.sequential();
  featureColumns: string[] = [];
  labelColumn = 'Apparent_Temperature';

  constructor(@Inject(PLATFORM_ID) private platformId: Object) {
    this.isBrowser = isPlatformBrowser(platformId);
  }

  ngAfterViewInit() {
    if (this.isBrowser) {
        this.loadData();
        this.visualizeDataset();
        this.linearRegression();
    }
  }

  async loadData() {
    const dataset = tf.data.csv(CSV_URL, {
      columnConfigs: {
        [this.labelColumn]: { isLabel: true }
      }
    });
    
    this.dataArray = await dataset.shuffle(1000).toArray();
    
    if (this.dataArray.length > 0) {
      const firstSample = this.dataArray[0];
      this.featureColumns = Object.keys(firstSample.xs).filter(
        name => name !== this.labelColumn
      );
      console.log('Feature columns:', this.featureColumns);
    }
    console.log(`Loaded ${this.dataArray.length} records`);
  }

  async visualizeDataset() {
    const points = this.dataArray.slice(0, 500); 
    
    const tempPoints = points.map(e => ({
      x: e.xs.Temperature,
      y: e.ys.Apparent_Temperature
    }));
    
    const humidityPoints = points.map(e => ({
      x: e.xs.Humidity,
      y: e.ys.Apparent_Temperature
    }));

    console.log('Sample tempPoints:', tempPoints.slice(0, 5));
    
    tfvis.render.scatterplot(
      { name: 'Temp vs Apparent Temp', tab: 'Charts' }, 
      { values: tempPoints }, 
      { xLabel: 'Temperature (C)', yLabel: 'Apparent Temperature (C)' }
    );

    tfvis.render.scatterplot(
      { name: 'Humidity vs Apparent Temp', tab: 'Charts' },
      { values: humidityPoints },
      { xLabel: 'Humidity (%)', yLabel: 'Apparent Temperature (C)' }
    );
  }

  async linearRegression() {
    if (this.dataArray.length === 0) {
      console.log("No data available for training");
      return;
    }
    
    const numberEpochs = 100;
    const batchSize = 32;
    
    const vectorizedData = this.dataArray.map(({xs, ys}) => ({
      xs: this.featureColumns.map(col => xs[col]),
      ys: [ys[this.labelColumn]]
    }));

    const splitIndex = Math.floor(0.8 * this.dataArray.length);
    const trainData = vectorizedData.slice(0, splitIndex);
    const valData = vectorizedData.slice(splitIndex);

    console.log(`Training on ${trainData.length} samples, validating on ${valData.length} samples`);
    
    const trainDataset = tf.data.array(trainData)
      .shuffle(trainData.length)
      .batch(batchSize);
    
    const valDataset = tf.data.array(valData).batch(batchSize);

    this.model = tf.sequential();
    this.model.add(tf.layers.dense({
      inputShape: [this.featureColumns.length],
      units: 1
    }));
    
    this.model.compile({
      optimizer: tf.train.adam(0.1),
      loss: 'meanSquaredError',
      metrics: ['mse']
    });

    //Use simplified metrics for TensorFlow.js compatibility
    await this.model.fitDataset(trainDataset, {
      epochs: numberEpochs,
      validationData: valDataset,
      callbacks: tfvis.show.fitCallbacks(
        { name: 'Training Performance', tab: 'Training' },
        ['loss', 'mse', 'val_loss', 'val_mse'],
        { height: 300, callbacks: ['onEpochEnd'] }
      )
    });
    
    console.log('Training completed');
  }
}