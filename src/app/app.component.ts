import { Component, Inject, PLATFORM_ID, AfterViewInit } from '@angular/core';
import { isPlatformBrowser } from '@angular/common';
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

const csvurl = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRm-d5gwpY6E-NYgp95ycNmQzPvQ8fAh5MgOI7Tn_Podim_OVBjn168oWAEQVSq2w/pub?gid=971307772&single=true&output=csv";

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements AfterViewInit {
  dataset: any;
  isBrowser: boolean;
  model= tf.sequential();

  constructor(@Inject(PLATFORM_ID) private platformId: Object) {
    this.isBrowser = isPlatformBrowser(platformId);
  }

  ngAfterViewInit() {
    if (this.isBrowser) {
      this.loadData();
      this.visualizeDataset();
      this.linearregression();
    }
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
    const shuffledDataset = this.dataset.shuffle(1000);
    const sampledDataset = await shuffledDataset.take(100).toArray();
    const dataset: any[] = [];
    const humidityDataset: any[] = [];

    sampledDataset.forEach((e:any) => {
      dataset.push({ x: e.xs.Temperature, y: e.ys.Apparent_Temperature });
      humidityDataset.push({ x: e.xs.Humidity, y: e.ys.Apparent_Temperature });
    });

    tfvis.render.scatterplot(
      { name: 'Temperature vs Apparent Temperature', tab: 'Charts' }, 
      { values: dataset }, 
      { xLabel: 'Temperature (C)', yLabel: 'Apparent Temperature (C)' }
    );

    tfvis.render.scatterplot(
      { name: 'Humidity vs Apparent Temperature', tab: 'Charts' },
      { values: humidityDataset },
      { xLabel: 'Humidity (%)', yLabel: 'Apparent Temperature (C)' }
    );
  }

  async linearregression() {
    const numberEpochs = 100;
    const columnNames = await this.dataset.columnNames();
    const featureColumns = columnNames.filter((name:any) => 
      name !== 'Apparent_Temperature' && name !== 'Humidity'
    );
    featureColumns.push('Humidity'); 
    const numOfFeatures = featureColumns.length;
    
    const shuffledDataset = this.dataset.shuffle(1000);
    const sampledDataset = await shuffledDataset.take(100).toArray();
    
    const features: number[][] = [];
    const target: number[] = [];
    
    sampledDataset.forEach((e:any) => {
      const row = featureColumns.map((col:any) => e.xs[col]);
      features.push(row);
      target.push(e.ys.Apparent_Temperature);
    });

    const featuresTensor = tf.tensor2d(features, [features.length, numOfFeatures]);
    const targetTensor = tf.tensor2d(target, [target.length, 1]);

    this.model = tf.sequential();
    this.model.add(tf.layers.dense({
      inputShape: [numOfFeatures],
      units: 1
    }));
    
    this.model.compile({
      optimizer: tf.train.adam(0.1),
      loss: 'meanSquaredError',
      metrics: ['mse']
    });

    // Training
    await this.model.fit(featuresTensor, targetTensor, {
      epochs: numberEpochs,
      validationSplit: 0.2,
      callbacks: tfvis.show.fitCallbacks(
        { name: 'Training Performance', tab: 'Training' },
        ['loss', 'mse', 'val_loss', 'val_mse'],
        { height: 300, callbacks: ['onEpochEnd'] }
      )
    });
  }
}