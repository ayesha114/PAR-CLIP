"""Fast, dataset-free checks; run with: python test_smoke.py"""
import unittest
import torch
from methods import unit, subspace, scores, train_parclip, train_zhang_style
from run import TinyCLIP

class TestConferenceCode(unittest.TestCase):
    def test_projection_and_fairness(self):
        torch.manual_seed(1)
        d = 16
        tr = unit(torch.randn(90,d))
        va = unit(torch.randn(45,d))
        y = torch.arange(9).repeat_interleave(10)
        vy = torch.arange(9).repeat_interleave(5)
        p = unit(torch.randn(9,d))
        o = unit(torch.randn(16,d))
        B, _, meta = subspace(tr, y, 5)
        self.assertEqual(tuple(B.shape), (d,5))
        self.assertEqual(meta['effective_k'],5)
        self.assertTrue(0. <= scores(va,p,o,vy)['macro_f1'] <= 1.)
        metrics, details, history, state = train_parclip(
            tr,y,va,vy,p,o,5,0.5,epochs=1,batch_size=27,seed=1)
        self.assertEqual(metrics['n'],45)
        self.assertEqual(details['effective_k'],5)
        self.assertEqual(len(history),1)
        self.assertIn('procrustes_R',state)

    def test_zhang_style_adapter(self):
        torch.manual_seed(2)
        x=unit(torch.randn(42,16))
        p=unit(torch.randn(7,16)); o=unit(torch.randn(16,16))
        y=torch.arange(7).repeat_interleave(6)
        metrics,history,_=train_zhang_style(x,y,x,y,p,o,epochs=1,batch_size=21)
        self.assertEqual(metrics['n'],42)
        self.assertEqual(len(history),1)

    def test_scratch_vlm(self):
        m=TinyCLIP(50)
        images=torch.randn(2,3,112,112)
        tokens=torch.randint(1,49,(2,30)); tokens[:,-5:]=0
        i,t=m(images,tokens)
        self.assertEqual(i.shape,(2,256))
        self.assertEqual(t.shape,(2,256))

if __name__=='__main__':
    unittest.main()
